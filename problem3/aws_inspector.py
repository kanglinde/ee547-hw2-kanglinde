
import sys
import argparse
import json
import re
from datetime import datetime, timezone
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, ConnectTimeoutError

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region")
    parser.add_argument("--output")
    parser.add_argument("--format", choices=["json", "table"], default="json")
    args = parser.parse_args()
    region = args.region if args.region else None
    output = args.output if args.output else None
    format = args.format
    return region, output, format

def format_datetime(time):
    return time.replace(microsecond=0).isoformat().replace("+00:00", "Z")

def verify_identity(region):
    sts = boto3.client('sts', region_name=region)
    try:
        identity = sts.get_caller_identity()
    except NoCredentialsError:
        print("[ERROR] AWS credentials not found")
        sys.exit(1)
    except Exception as e:
        # error occurs when trying to connect "https://sts.{invalid_region}.amazonaws.com/"
        if re.match(r'Could not connect to the endpoint URL', str(e)):
            print(f'[ERROR] Invalid region: {region}')
        else:
            print(f'[ERROR] {e}')
        sys.exit(1)
    return identity

def iam_user(region):
    users = []
    iam = boto3.client('iam', region_name=region)
    try:
        # use pagination in case #page > 1
        paginator = iam.get_paginator('list_users')
        for page in paginator.paginate():
            for user in page['Users']:
                # get_user() provides more details
                user_details = iam.get_user(UserName=user['UserName'])['User']
                user_policies = iam.list_attached_user_policies(UserName=user['UserName'])['AttachedPolicies']
                user_data = {
                    "username": user_details['UserName'],
                    "user_id": user_details['UserId'],
                    "arn": user_details['Arn'],
                    "create_date": format_datetime(user_details['CreateDate']),
                    "last_activity": format_datetime(user_details['PasswordLastUsed']) if 'PasswordLastUsed' in user_details else '-',
                    "attached_policies": [{
                        "policy_name": policy['PolicyName'],
                        "policy_arn": policy['PolicyArn']
                    } for policy in user_policies]
                }
                users.append(user_data)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f'[ERROR] {error_code}: {error_message}')
    except Exception as e:
        print(f'[ERROR] {e}')
    return users

def ec2_inst(region):
    instances = []
    security = []
    ec2 = boto3.client('ec2', region_name=region)
    try:
        paginator = ec2.get_paginator('describe_instances')
        for page in paginator.paginate():
            for reserve in page['Reservations']:
                for inst in reserve['Instances']:
                    inst_data = {
                        'instance_id': inst['InstanceId'],
                        'instance_type': inst['InstanceType'],
                        'state': inst['State']['Name'],
                        'public_ip': inst.get('PublicIpAddress', '-'),
                        'private_ip': inst['PrivateIpAddress'],
                        'availability_zone': inst['Placement']['AvailabilityZone'],
                        'launch_time': format_datetime(inst['LaunchTime']),
                        'ami_id': inst['ImageId'],
                        'ami_name': ec2.describe_images(ImageIds=[inst['ImageId']])['Images'][0]['Description'],
                        'security_groups': []
                    }

                    security_groups = ec2.describe_security_groups()['SecurityGroups']
                    for group in security_groups:
                        inst_data['security_groups'].append(group['GroupId'])
                        group_data = {
                            'group_id': group['GroupId'],
                            'group_name': group['GroupName'],
                            'description': group['Description'],
                            'vpc_id': group['VpcId']
                        }
                        inbound_rules = []
                        for rule in group['IpPermissions']:
                            protocol = rule['IpProtocol'] if rule['IpProtocol'] != '-1' else "all"
                            range = f'{rule["FromPort"]}-{rule["ToPort"]}' if 'FromPort' in rule and 'ToPort' in rule else "all"
                            src = rule['IpRanges'][0]['CidrIp'] if len(rule['IpRanges']) > 0 else ""
                            inbound_rules.append({
                                "protocol": protocol,
                                "port_range": range,
                                "source": src
                            })
                        group_data['inbound_rules'] = inbound_rules
                        outbound_rules = []
                        for rule in group['IpPermissionsEgress']:
                            protocol = rule['IpProtocol'] if rule['IpProtocol'] != '-1' else "all"
                            range = f'{rule["FromPort"]}-{rule["ToPort"]}' if 'FromPort' in rule and 'ToPort' in rule else "all"
                            dst = rule['IpRanges'][0]['CidrIp'] if len(rule['IpRanges']) > 0 else ""
                            outbound_rules.append({
                                "protocol": protocol,
                                "port_range": range,
                                "destination": dst
                            })
                        group_data['outbound_rules'] = outbound_rules
                        security.append(group_data)

                    tags = {}
                    for tag in inst['Tags']:
                        tags[tag['Key']] = tag['Value']
                    inst_data['tags'] = tags
                    instances.append(inst_data)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f'[ERROR] {error_code}: {error_message}')
    except Exception as e:
        print(f'[ERROR] {e}')
    return instances, security

def s3_buk(region):
    buckets = []
    s3 = boto3.client('s3', region_name=region)
    try:
        bucket_list = s3.list_buckets()
        for bucket in bucket_list['Buckets']:
            buck_data = {
                'bucket_name': bucket['Name'],
                'creation_date': format_datetime(bucket['CreationDate'])
            }
            location = s3.get_bucket_location(Bucket=bucket['Name'])['LocationConstraint']
            buck_data['region'] = location if location is not None else 'us-east-1'
            buck_data['object_count'] = 0
            buck_data['size_bytes'] = 0
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket['Name']):
                for obj in page.get('Contents', []):
                    buck_data['object_count'] += 1
                    buck_data['size_bytes'] += obj['Size']
            buckets.append(buck_data)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        print(f'[ERROR] {error_code}: {error_message}')
    except Exception as e:
        print(f'[ERROR] {e}')
    return buckets

def make_table(data):
    table = []

    table.append(f'AWS Account: {data["account_info"]["account_id"]} ({data["account_info"]["region"]})')
    table.append(f'Scan Time: {data["account_info"]["scan_timestamp"].replace("T", " ").replace("Z", " UTC")}')
    table.append("")

    table.append(f'IAM USERS ({data["summary"]["total_users"]} total)')
    table.append(f'{"Username":<20} {"Create Date":<15} {"Last Activity":<15} {"Policies":<10}')
    for user in data['resources']['iam_users']:
        create_date = "-" if user["create_date"].find("T") == -1 else user["create_date"][:user["create_date"].find("T")]
        last_activity = "-" if user["last_activity"].find("T") == -1 else user["last_activity"][:user["last_activity"].find("T")]
        table.append(f'{user["username"]:<20} {create_date:<15} {last_activity:<15} {len(user["attached_policies"]):<10}')
    table.append("")

    table.append(f'EC2 INSTANCES ({data["summary"]["running_instances"]} running, {len(data["resources"]["ec2_instances"]) - data["summary"]["running_instances"]} stopped)')
    table.append(f'{"Instance ID":<20} {"Type":<10} {"State":<10} {"Public IP":<15} {"Launch Time":<15}')
    for inst in data['resources']['ec2_instances']:
        table.append(f'{inst["instance_id"]:<20} {inst["instance_type"]:<10} {inst["state"]:<10} {inst["public_ip"]:<15} {inst["launch_time"].replace("T", " ").replace("Z", ""):<15}')
    table.append("")

    table.append(f'S3 BUCKETS ({data["summary"]["total_buckets"]} total)')
    table.append(f'{"Bucket Name":<20} {"Region":<15} {"Created":<15} {"Objects":<10} {"Size (MB)":<15}')
    for buck in data['resources']['s3_buckets']:
        creation_date = "-" if buck["creation_date"].find("T") == -1 else buck["creation_date"][:buck["creation_date"].find("T")]
        table.append(f'{buck["bucket_name"]:<20} {buck["region"]:<15} {creation_date:<15} {buck["object_count"]:<10} ~{float(buck["size_bytes"])/1024:<15.1f}')
    table.append("")

    table.append(f'SECURITY GROUPS ({data["summary"]["security_groups"]} total)')
    table.append(f'{"Group ID":<25} {"Name":<20} {"VPC ID":<25} {"Inbound Rules":<10}')
    for group in data['resources']['security_groups']:
        table.append(f'{group["group_id"]:<25} {group["group_name"]:<20} {group["vpc_id"]:<25} {len(group["inbound_rules"]):<10}')

    return table

def main():
    region, output, format = parse_arg()

    # if region not provided, use default from credentials/environment
    if region is None:
        session = boto3.session.Session()
        region = session.region_name

    # Verify authentication and validate region
    identity = verify_identity(region)

    # Collect account info
    account_info = {
        "account_id": identity["Account"],
        "user_arn": identity["Arn"],
        "region": region,
        "scan_timestamp": format_datetime(datetime.now(timezone.utc))
    }

    # Collect Resource
    iam_users = iam_user(region)
    if len(iam_users) == 0: print(f'[WARNING] No IAM users found in {region}', flush=True)
    ec2_instances, security_groups = ec2_inst(region)
    if len(ec2_instances) == 0: print(f'[WARNING] No EC2 instances found in {region}', flush=True)
    s3_buckets = s3_buk(region)
    if len(s3_buckets) == 0: print(f'[WARNING] No S3 buckets found in {region}', flush=True)
    resources = {
        "iam_users": iam_users,
        "ec2_instances": ec2_instances,
        "s3_buckets": s3_buckets,
        "security_groups": security_groups
    }

    # Generate summary
    summary = {
        "total_users": len(resources["iam_users"]),
        "running_instances": len([inst for inst in resources["ec2_instances"] if inst["state"] == "running"]),
        "total_buckets": len(resources["s3_buckets"]),
        "security_groups": len(resources["security_groups"])
    }

    # Collect Output
    data = {
        "account_info": account_info,
        "resources": resources,
        "summary": summary
    }
    table = make_table(data)

    # Print output
    if output is None:
        # stdout
        if format == "table":
            for line in table:
                print(line, flush=True)
        else: #json
            print(json.dumps(data, indent=2))
    else:
        with open(output, 'w') as file:
            if format == "table":
                for line in table:
                    file.write(line+'\n')
            else:
                json.dump(data, file, indent=2)

if __name__ == "__main__":
    main()