{{/* Chart name. */}}
{{- define "volforecast.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Fully qualified resource name prefix. */}}
{{- define "volforecast.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{- define "volforecast.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Standard labels (no component). */}}
{{- define "volforecast.labels" -}}
helm.sh/chart: {{ include "volforecast.chart" . }}
app.kubernetes.io/name: {{ include "volforecast.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: volforecast
{{- end }}

{{/* Selector labels for the API pods. */}}
{{- define "volforecast.apiSelectorLabels" -}}
app.kubernetes.io/name: {{ include "volforecast.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: api
{{- end }}

{{/* Selector labels for the job pods. */}}
{{- define "volforecast.jobSelectorLabels" -}}
app.kubernetes.io/name: {{ include "volforecast.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: job
{{- end }}

{{- define "volforecast.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "volforecast.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "volforecast.retrainServiceAccountName" -}}
{{- printf "%s-retrain" (include "volforecast.fullname" .) }}
{{- end }}

{{/* Secret consumed through envFrom. */}}
{{- define "volforecast.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- printf "%s-secrets" (include "volforecast.fullname" .) }}
{{- end }}
{{- end }}

{{- define "volforecast.configMapName" -}}
{{- printf "%s-config" (include "volforecast.fullname" .) }}
{{- end }}

{{- define "volforecast.dataClaimName" -}}
{{- default (printf "%s-data" (include "volforecast.fullname" .)) .Values.dataVolume.existingClaim }}
{{- end }}

{{/* Image reference: call with (dict "repo" <repository> "root" $). */}}
{{- define "volforecast.image" -}}
{{- printf "%s:%s" .repo (default .root.Chart.AppVersion .root.Values.image.tag) }}
{{- end }}

{{/* Data + tmp volumes shared by API and jobs. */}}
{{- define "volforecast.volumes" -}}
- name: tmp
  emptyDir:
    sizeLimit: 512Mi
- name: data
  {{- if eq .Values.dataVolume.type "pvc" }}
  persistentVolumeClaim:
    claimName: {{ include "volforecast.dataClaimName" . }}
  {{- else }}
  emptyDir:
    sizeLimit: {{ .Values.dataVolume.sizeLimit }}
  {{- end }}
{{- end }}

{{- define "volforecast.volumeMounts" -}}
- name: tmp
  mountPath: /tmp
- name: data
  mountPath: {{ index .Values.config "VOLF_DATA_DIR" | default "/data" }}
{{- end }}
