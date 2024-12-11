import boto3
import os
import sys
from submodules.threshold_config import ThresholdConfig
from submodules.video_analyzer import VideoAnalyzer
from submodules.video_processor import adjust_thresholds_cli, process_video

# Redirect configuration directories
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def lambda_handler(event, context):
    # Initialize S3 client
    s3_client = boto3.client(
        's3',
        aws_access_key_id=os.getenv("ACCESS_KEY"),
        aws_secret_access_key=os.getenv("SECRET_KEY"),
        region_name=os.getenv("REGION_NAME")
    )
    # Extract bucket name and video file key from the event
    try:
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        video_key = event['Records'][0]['s3']['object']['key']
    except (KeyError, IndexError) as e:
        return {
            'statusCode': 400,
            'body': f'Invalid S3 event data: {str(e)}'
        }
    
    try:
        # Create necessary directories
        input_dir = '/tmp/input_videos'
        output_dir = '/tmp/output'
        ensure_directory_exists(input_dir)
        ensure_directory_exists(output_dir)
        
        # Download the video file
        download_path = os.path.join(input_dir, os.path.basename(video_key))
        print(f"Attempting to download {video_key} to {download_path}")
        s3_client.download_file(bucket_name, video_key, download_path)
        print(f"Successfully downloaded video to {download_path}")
        
        # Initialize configuration
        config = ThresholdConfig()
        
        # Create analyzer
        analyzer = VideoAnalyzer(download_path, config)
        
        # Analyze video
        analyzer.analyze_video()
        
        # Adjust thresholds through CLI
        print("\nStarting threshold adjustment interface...")
        thresholds = adjust_thresholds_cli(analyzer)
        
        # Process the video with final thresholds
        process_video(download_path, output_dir, thresholds,s3_bucket="flipkartdataset1",class_name=video_key.split(".")[0])
        
        
        return {
            'statusCode': 200,
            'body': {
                'message': f"Successfully processed video: {video_key}",
            }
        }
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error processing video: {str(e)}"
        }