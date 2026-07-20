resource "aws_efs_file_system" "main" {
  creation_token = "${local.name_prefix}-efs"
  encrypted      = true

  lifecycle_policy {
    transition_to_ia = "AFTER_30_DAYS"
  }

  tags = { Name = "${local.name_prefix}-efs" }
}

resource "aws_efs_mount_target" "main" {
  count           = length(aws_subnet.private)
  file_system_id  = aws_efs_file_system.main.id
  subnet_id       = aws_subnet.private[count.index].id
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "cv_artifacts" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/cv-artifacts"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = { Name = "${local.name_prefix}-ap-cv" }
}

resource "aws_efs_access_point" "mlair_models" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/mlair-models"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = { Name = "${local.name_prefix}-ap-models" }
}

resource "aws_efs_access_point" "mlair_datasets" {
  file_system_id = aws_efs_file_system.main.id

  posix_user {
    gid = 1000
    uid = 1000
  }

  root_directory {
    path = "/mlair-datasets"
    creation_info {
      owner_gid   = 1000
      owner_uid   = 1000
      permissions = "755"
    }
  }

  tags = { Name = "${local.name_prefix}-ap-datasets" }
}
