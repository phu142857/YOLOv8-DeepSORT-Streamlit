resource "aws_lb" "main" {
  name               = "${local.name_prefix}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  tags = { Name = "${local.name_prefix}-alb" }
}

resource "aws_lb_target_group" "hub" {
  name     = "${local.name_prefix}-hub"
  port     = 3000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path                = "/"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    interval            = 30
  }
}

resource "aws_lb_target_group" "mlair_api" {
  name     = "${local.name_prefix}-mlair-api"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path = "/health"
  }
}

resource "aws_lb_target_group" "mlair_realtime" {
  name     = "${local.name_prefix}-mlair-rt"
  port     = 8001
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path = "/healthz"
  }
}

resource "aws_lb_target_group" "cv_api" {
  name     = "${local.name_prefix}-cv-api"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path = "/health"
  }
}

resource "aws_lb_target_group" "cv_ui" {
  name     = "${local.name_prefix}-cv-ui"
  port     = 8501
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id

  health_check {
    path = "/_stcore/health"
  }
}

resource "aws_lb_target_group_attachment" "hub" {
  target_group_arn = aws_lb_target_group.hub.arn
  target_id        = aws_instance.app.id
  port             = 3000
}

resource "aws_lb_target_group_attachment" "mlair_api" {
  target_group_arn = aws_lb_target_group.mlair_api.arn
  target_id        = aws_instance.app.id
  port             = 8080
}

resource "aws_lb_target_group_attachment" "mlair_realtime" {
  target_group_arn = aws_lb_target_group.mlair_realtime.arn
  target_id        = aws_instance.app.id
  port             = 8001
}

resource "aws_lb_target_group_attachment" "cv_api" {
  target_group_arn = aws_lb_target_group.cv_api.arn
  target_id        = aws_instance.app.id
  port             = 8000
}

resource "aws_lb_target_group_attachment" "cv_ui" {
  target_group_arn = aws_lb_target_group.cv_ui.arn
  target_id        = aws_instance.app.id
  port             = 8501
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.hub.arn
  }
}

resource "aws_lb_listener_rule" "mlair_api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 10

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlair_api.arn
  }

  condition {
    path_pattern {
      values = ["/v1/*", "/health", "/docs", "/openapi.json"]
    }
  }
}

resource "aws_lb_listener_rule" "mlair_realtime" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 15

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.mlair_realtime.arn
  }

  condition {
    path_pattern {
      values = ["/ws", "/ws/*"]
    }
  }
}

resource "aws_lb_listener_rule" "cv_api" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 20

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.cv_api.arn
  }

  condition {
    path_pattern {
      values = ["/cv-api/*"]
    }
  }
}

resource "aws_lb_listener_rule" "cv_ui" {
  listener_arn = aws_lb_listener.http.arn
  priority     = 30

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.cv_ui.arn
  }

  condition {
    path_pattern {
      values = ["/cv", "/cv/*"]
    }
  }
}
