terraform {
  required_version = ">= 1.6"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Local state is fine for a personal project.
  # To share state (optional): add an S3 backend here.
}

provider "aws" {
  region = var.region
}

# ---------------------------------------------------------------------------
# Look up a pre-allocated EIP when an allocation ID is supplied
# ---------------------------------------------------------------------------

data "aws_eip" "persistent" {
  count = var.eip_allocation_id != "" ? 1 : 0
  id    = var.eip_allocation_id
}

# ---------------------------------------------------------------------------
# Fetch latest Amazon Linux 2023 ARM64 AMI
# ---------------------------------------------------------------------------

data "aws_ami" "al2023_arm" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-arm64"]
  }

  filter {
    name   = "architecture"
    values = ["arm64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ---------------------------------------------------------------------------
# SSH key pair
# ---------------------------------------------------------------------------

resource "aws_key_pair" "bot" {
  key_name   = "${var.name_prefix}-key"
  public_key = var.ssh_public_key
}

# ---------------------------------------------------------------------------
# Security group — SSH in, all outbound
# ---------------------------------------------------------------------------

resource "aws_security_group" "bot" {
  name        = "${var.name_prefix}-sg"
  description = "Allow SSH inbound, all outbound for the grid bot"

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.name_prefix}-sg"
  }
}

# ---------------------------------------------------------------------------
# EC2 instance — t4g.nano (ARM Graviton2)
# ---------------------------------------------------------------------------

resource "aws_instance" "bot" {
  ami                    = data.aws_ami.al2023_arm.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.bot.key_name
  vpc_security_group_ids = [aws_security_group.bot.id]

  # Bootstrap: install Python 3.11+, git, clone repo, set up venv + systemd
  user_data = templatefile("${path.module}/user_data.sh", {
    repo_url    = var.repo_url
    repo_branch = var.repo_branch
  })

  # AL2023 AMI snapshot requires >= 30 GB
  root_block_device {
    volume_type = "gp3"
    volume_size = 30
  }

  tags = {
    Name = var.name_prefix
  }

  lifecycle {
    # Prevent accidental replacement if AMI updates
    ignore_changes = [ami, user_data]
  }
}

# ---------------------------------------------------------------------------
# Elastic IP — either associate a pre-allocated EIP (for stable IP whitelisting)
# or create a new one for ad-hoc use.
# ---------------------------------------------------------------------------

resource "aws_eip" "bot" {
  count    = var.eip_allocation_id == "" ? 1 : 0
  instance = aws_instance.bot.id
  domain   = "vpc"

  tags = {
    Name = "${var.name_prefix}-eip"
  }
}

resource "aws_eip_association" "bot" {
  count         = var.eip_allocation_id != "" ? 1 : 0
  allocation_id = var.eip_allocation_id
  instance_id   = aws_instance.bot.id
}
