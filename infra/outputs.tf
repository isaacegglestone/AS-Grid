output "instance_ip" {
  description = "Public IP of the bot instance"
  value       = var.eip_allocation_id != "" ? data.aws_eip.persistent[0].public_ip : aws_eip.bot[0].public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.bot.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh ec2-user@${var.eip_allocation_id != "" ? data.aws_eip.persistent[0].public_ip : aws_eip.bot[0].public_ip}"
}
