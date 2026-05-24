data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

resource "aws_instance" "app" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.node_instance_type
  subnet_id              = aws_subnet.private[0].id
  vpc_security_group_ids = [aws_security_group.app.id]
  iam_instance_profile   = aws_iam_instance_profile.app.name
  key_name               = var.key_name != "" ? var.key_name : null

  root_block_device {
    volume_size = var.app_volume_size_gb
    volume_type = "gp3"
    encrypted   = true
  }

  user_data_base64 = base64encode(templatefile("${path.module}/templates/user_data.sh.tpl", {
    efs_id       = aws_efs_file_system.main.id
    region       = var.aws_region
    ap_cv        = aws_efs_access_point.cv_artifacts.id
    ap_models    = aws_efs_access_point.mlair_models.id
    ap_datasets  = aws_efs_access_point.mlair_datasets.id
    secrets_arn  = aws_secretsmanager_secret.app.arn
    ecr_registry = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
    name_prefix  = local.name_prefix
  }))

  tags = { Name = "${local.name_prefix}-app" }
}
