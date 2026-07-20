{{- define "mlair-stack.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mlair-stack.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- printf "%s" $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "mlair-stack.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "mlair-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "mlair-stack.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mlair-stack.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "mlair-stack.serviceName" -}}
{{- $base := include "mlair-stack.fullname" . -}}
{{- printf "%s-%s" $base .service | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "mlair-stack.mlairApiImage" -}}
{{- if .Values.mlairApi.image -}}
{{- .Values.mlairApi.image -}}
{{- else -}}
{{- printf "%s/%s/ml-air-api-cv-workload:%s" .Values.global.ecrRegistry .Values.global.namePrefix .Values.global.imageTag -}}
{{- end -}}
{{- end -}}

{{- define "mlair-stack.cvWorkloadImage" -}}
{{- printf "%s/%s/cv-lifecycle-workload:%s" .Values.global.ecrRegistry .Values.global.namePrefix .Values.global.imageTag -}}
{{- end -}}

{{- define "mlair-stack.mlairCoreImage" -}}
{{- printf "%s/%s:%s" .Values.global.mlairRegistry .name .Values.global.mlairImageTag -}}
{{- end -}}

