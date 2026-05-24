# RDS master password: no / @ " or space (AWS InvalidParameterValue).
resource "random_password" "db" {
  length           = 32
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?"
}

resource "random_password" "jwt" {
  length  = 48
  special = false
}

resource "random_password" "tracking" {
  length  = 32
  special = false
}

resource "aws_secretsmanager_secret" "app" {
  name                    = "${local.name_prefix}/app-env"
  # dev/staging: immediate delete on destroy so redeploy can reuse the name
  recovery_window_in_days = var.environment == "prod" ? 30 : 0
}

resource "aws_secretsmanager_secret_version" "app" {
  secret_id = aws_secretsmanager_secret.app.id
  secret_string = jsonencode({
    POSTGRES_PASSWORD          = random_password.db.result
    ML_AIR_JWT_HS256_SECRET    = random_password.jwt.result
    ML_AIR_TRACKING_TOKEN      = random_password.tracking.result
    CV_MLAIR_TOKEN             = random_password.tracking.result
    ML_AIR_MANIFEST_SIGNING_KEY = random_password.jwt.result
  })
}
