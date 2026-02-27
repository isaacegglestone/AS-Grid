output "instance_ip" {
  description = "Elastic IP of the bot instance — add this as GitHub secret EC2_HOST"
  value       = aws_eip.bot.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.bot.id
}

output "ssh_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh ec2-user@${aws_eip.bot.public_ip}"
}
