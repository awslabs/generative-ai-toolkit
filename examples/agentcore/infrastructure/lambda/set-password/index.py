import json
import logging

import boto3

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def on_event(event, context):
    """
    Custom resource handler for setting Cognito user password
    """
    logger.info(f"Event: {json.dumps(event, default=str)}")

    request_type = event["RequestType"]
    if request_type in {"Create", "Update"}:
        return on_create_or_update(event, context)
    elif request_type == "Delete":
        return on_delete(event, context)
    else:
        raise Exception(f"Invalid request type: {request_type}")


def on_create_or_update(event, context):
    """
    Handle Create and Update events
    """
    props = event["ResourceProperties"]
    user_pool_id = props["UserPoolId"]
    username = props["Username"]
    password = props["Password"]

    # Convert string 'true'/'false' to boolean
    permanent_str = props.get("Permanent", "true")
    permanent = (
        permanent_str.lower() == "true"
        if isinstance(permanent_str, str)
        else bool(permanent_str)
    )

    cognito = boto3.client("cognito-idp")

    try:
        # Set the user password as permanent
        cognito.admin_set_user_password(
            UserPoolId=user_pool_id,
            Username=username,
            Password=password,
            Permanent=permanent,
        )

        logger.info(
            f"Successfully set password for user {username} in pool {user_pool_id}"
        )

        return {
            "PhysicalResourceId": f"{user_pool_id}-{username}",
            "Data": {"UserPoolId": user_pool_id, "Username": username},
        }
    except Exception as e:
        logger.error(f"Error setting password: {str(e)}")
        raise e


def on_delete(event, context):
    """
    Handle Delete events - nothing to do for password setting
    """
    logger.info("Delete event - no action required for password setting")
    return {"PhysicalResourceId": event["PhysicalResourceId"]}
