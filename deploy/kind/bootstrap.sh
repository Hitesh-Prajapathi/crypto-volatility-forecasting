#!/usr/bin/env bash
# Bootstrap a local kind cluster with the volforecast platform stack.
#
#   deploy/kind/bootstrap.sh                      # create cluster + platform + root app
#   REPO_URL=https://github.com/me/repo.git deploy/kind/bootstrap.sh
#   SKIP_BUILD=1 deploy/kind/bootstrap.sh         # reuse already built local images
#
# Requires: docker, kind, kubectl, helm (and optionally kubeseal). Not run in CI.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CLUSTER_NAME="${CLUSTER_NAME:-volforecast}"
NAMESPACE="${NAMESPACE:-volforecast}"
IMAGE_TAG="${IMAGE_TAG:-latest}" # must match image.tag in values-kind.yaml
IMAGE_API="${IMAGE_API:-ghcr.io/OWNER/crypto-volatility-forecasting-api}"
IMAGE_JOBS="${IMAGE_JOBS:-ghcr.io/OWNER/crypto-volatility-forecasting-jobs}"
REPO_URL="${REPO_URL:-}"

# Pinned chart versions.
INGRESS_NGINX_VERSION="4.15.1"
METRICS_SERVER_VERSION="3.14.0"
SEALED_SECRETS_VERSION="2.22.0"
KUBE_PROMETHEUS_STACK_VERSION="93.3.0"
ARGOCD_VERSION="10.18.0"

# Dev-only credentials (match .env.example); override via environment.
DEV_DATABASE_URL="${DEV_DATABASE_URL:-postgresql+psycopg://volf:volf-dev-only@postgres:5432/volforecast}"
DEV_S3_KEY="${DEV_S3_KEY:-minioadmin}"
DEV_S3_SECRET="${DEV_S3_SECRET:-minioadmin-dev-only}"

log() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required tool: $1" >&2; exit 1; }; }

for tool in docker kind kubectl helm; do need "$tool"; done

log "Creating kind cluster '${CLUSTER_NAME}'"
if kind get clusters | grep -qx "${CLUSTER_NAME}"; then
  echo "cluster already exists, reusing"
else
  kind create cluster --config "${ROOT}/deploy/kind/cluster.yaml"
fi
kubectl config use-context "kind-${CLUSTER_NAME}"

log "Adding Helm repositories"
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add metrics-server https://kubernetes-sigs.github.io/metrics-server/
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update

log "Installing ingress-nginx ${INGRESS_NGINX_VERSION}"
helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
  --version "${INGRESS_NGINX_VERSION}" --namespace ingress-nginx --create-namespace \
  --set controller.hostPort.enabled=true \
  --set controller.service.type=NodePort \
  --set controller.publishService.enabled=false \
  --set controller.nodeSelector."ingress-ready"=true \
  --set "controller.tolerations[0].key=node-role.kubernetes.io/control-plane" \
  --set "controller.tolerations[0].operator=Equal" \
  --set "controller.tolerations[0].effect=NoSchedule" \
  --wait --timeout 5m

log "Installing metrics-server ${METRICS_SERVER_VERSION} (kubelet-insecure-tls for kind)"
helm upgrade --install metrics-server metrics-server/metrics-server \
  --version "${METRICS_SERVER_VERSION}" --namespace kube-system \
  --set "args={--kubelet-insecure-tls}" \
  --wait --timeout 5m

log "Installing sealed-secrets ${SEALED_SECRETS_VERSION}"
helm upgrade --install sealed-secrets oci://registry-1.docker.io/bitnamicharts/sealed-secrets \
  --version "${SEALED_SECRETS_VERSION}" --namespace kube-system \
  --set fullnameOverride=sealed-secrets-controller \
  --wait --timeout 5m

log "Installing kube-prometheus-stack ${KUBE_PROMETHEUS_STACK_VERSION}"
# Same values Argo CD uses for the platform-monitoring app (deploy/argocd/apps).
helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --version "${KUBE_PROMETHEUS_STACK_VERSION}" --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.ruleSelectorNilUsesHelmValues=false \
  --set grafana.sidecar.dashboards.enabled=true \
  --set grafana.sidecar.dashboards.searchNamespace=ALL \
  --wait --timeout 10m

log "Installing Argo CD ${ARGOCD_VERSION}"
helm upgrade --install argocd argo/argo-cd \
  --version "${ARGOCD_VERSION}" --namespace argocd --create-namespace \
  --wait --timeout 10m

log "Loading local images into kind (set SKIP_BUILD=1 to skip building)"
if [[ -z "${SKIP_BUILD:-}" ]]; then
  docker build -f "${ROOT}/docker/api.Dockerfile" -t "${IMAGE_API}:${IMAGE_TAG}" "${ROOT}"
  docker build -f "${ROOT}/docker/jobs.Dockerfile" -t "${IMAGE_JOBS}:${IMAGE_TAG}" "${ROOT}"
fi
kind load docker-image "${IMAGE_API}:${IMAGE_TAG}" "${IMAGE_JOBS}:${IMAGE_TAG}" --name "${CLUSTER_NAME}"

log "Creating the application secret (${NAMESPACE}/volforecast-secrets)"
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
secret_yaml="$(kubectl create secret generic volforecast-secrets --namespace "${NAMESPACE}" \
  --from-literal=VOLF_DATABASE_URL="${DEV_DATABASE_URL}" \
  --from-literal=AWS_ACCESS_KEY_ID="${DEV_S3_KEY}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${DEV_S3_SECRET}" \
  --dry-run=client -o yaml)"
if command -v kubeseal >/dev/null 2>&1; then
  echo "kubeseal found: applying a SealedSecret (safe to commit the sealed output)"
  printf '%s\n' "${secret_yaml}" |
    kubeseal --controller-name sealed-secrets-controller --controller-namespace kube-system --format yaml |
    kubectl apply -f -
else
  echo "kubeseal not found: applying a plain dev Secret directly (never committed to git)"
  printf '%s\n' "${secret_yaml}" | kubectl apply -f -
fi

log "Applying the Argo CD root application"
if [[ -n "${REPO_URL}" ]]; then
  sed "s#https://github.com/OWNER/crypto-volatility-forecasting.git#${REPO_URL}#g" \
    "${ROOT}/deploy/argocd/root-app.yaml" | kubectl apply -f -
else
  echo "REPO_URL not set: applying with the placeholder repoURL (edit deploy/argocd/*.yaml first)"
  kubectl apply -f "${ROOT}/deploy/argocd/root-app.yaml"
fi

cat <<EOF

Done. Next steps:
  Argo CD UI : kubectl -n argocd port-forward svc/argocd-server 8080:443
  admin pass : kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath='{.data.password}' | base64 -d
  API        : curl http://localhost:30080/healthz   (or http://volforecast.localtest.me/healthz)
  Grafana    : kubectl -n monitoring port-forward svc/kube-prometheus-stack-grafana 3000:80
Tear down    : kind delete cluster --name ${CLUSTER_NAME}
EOF
