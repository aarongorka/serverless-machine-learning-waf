#!/usr/bin/env python3.6
import os
import logging
import aws_lambda_logging
import json
import uuid
from dateutil.tz import tzlocal
from dateutil.tz import tzutc

aws_lambda_logging.setup(level=os.environ.get('LOGLEVEL', 'INFO'), env=os.environ.get('ENV'))
logging.info(json.dumps({'message': 'initialising'}))
aws_lambda_logging.setup(level=os.environ.get('LOGLEVEL', 'INFO'), env=os.environ.get('ENV'))


def handler(event, context):
    """Handler for serverless-machine-learning-waf"""
    correlation_id = get_correlation_id(event=event)
    aws_lambda_logging.setup(level=os.environ.get('LOGLEVEL', 'INFO'), env=os.environ.get('ENV'), correlation_id=correlation_id)

    try:
        logging.debug(json.dumps({'message': 'logging event', 'status': 'success', 'event': event}))
    except:
        logging.exception(json.dumps({'message': 'logging event', 'status': 'failed'}))
        raise

    try:
        # do a thing
        thing = event
        logging.debug(json.dumps({'message': 'thing', 'status': 'success', 'thing': thing}))
    except:
        logging.exception(json.dumps({"message": "thing", "status": "failed"}))
        response = {
            "statusCode": 503,
            'headers': {
                'Content-Type': 'application/json',
            }
        }
        return response

    response = {
        "statusCode": 200,
        "body": json.dumps(thing),
        'headers': {
            'Content-Type': 'application/json',
        }
    }
    logging.info(json.dumps({'message': 'responding', 'response': response}))
    return response


def get_correlation_id(body=None, payload=None, event=None):
    correlation_id = None
    if event:
        try:
            correlation_id = event['headers']['X-Amzn-Trace-Id'].split('=')[1]
        except:
            pass

    if body:
        try:
            correlation_id = body['trigger_id'][0]
        except:
            pass
    elif payload:
        try:
            correlation_id = payload['trigger_id']
        except:
            pass

    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    return correlation_id
