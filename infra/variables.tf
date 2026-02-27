variable "region" {
  description = "AWS region to deploy into (ap-southeast-1 = Singapore, closest to most Asian exchanges)"
  type        = string
  default     = "ap-southeast-1"
}

variable "instance_type" {
  description = "EC2 instance type. t4g.nano (~$3.80/mo) is sufficient for a single Python bot."
  type        = string
  default     = "t4g.nano"
}

variable "ssh_public_key" {
  description = <<-EOT
    Contents of your SSH public key (~/.ssh/id_ed25519.pub or similar).
    Generate one with: ssh-keygen -t ed25519 -C "bitunix-bot"
    Then add the private key to GitHub secret EC2_SSH_KEY.
  EOT
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed to SSH to the instance. Defaults to your own IP — get it from https://checkip.amazonaws.com"
  type        = string
  default     = "0.0.0.0/0"  # Tighten this to your IP for production: "1.2.3.4/32"
}

variable "repo_url" {
  description = "GitHub repo URL to clone onto the instance"
  type        = string
  default     = "https://github.com/isaacegglestone/AS-Grid.git"
}

variable "repo_branch" {
  description = "Branch to deploy"
  type        = string
  default     = "main"
}
