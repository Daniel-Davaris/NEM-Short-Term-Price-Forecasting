#!/usr/bin/env bash
# s3_sync.sh — sync parquet data directories between local and S3
#
# Usage:
#   ./s3_sync.sh setup   — create the S3 bucket (run once)
#   ./s3_sync.sh push    — upload local parquets to S3
#   ./s3_sync.sh pull    — download parquets from S3 to local

set -euo pipefail

BUCKET="s3-forecasting"
REGION="ap-southeast-2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

do_sync() {
    local mode="$1"
    local local_dir="$2"
    local s3_key="$3"
    local local_path="$SCRIPT_DIR/$local_dir"
    local s3_path="s3://$BUCKET/$s3_key"

    if [[ "$mode" == "push" ]]; then
        echo "  → $local_dir"
        aws s3 sync "$local_path" "$s3_path" \
            --region "$REGION" \
            --exclude "*" --include "*.parquet"
    else
        echo "  ← $local_dir"
        mkdir -p "$local_path"
        aws s3 sync "$s3_path" "$local_path" \
            --region "$REGION" \
            --exclude "*" --include "*.parquet"
    fi
}

case "${1:-}" in
    setup)
        echo "Creating bucket s3://$BUCKET in $REGION ..."
        aws s3api create-bucket \
            --bucket "$BUCKET" \
            --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
        aws s3api put-public-access-block \
            --bucket "$BUCKET" \
            --public-access-block-configuration \
                BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
        echo "Done. Bucket s3://$BUCKET is ready."
        ;;
    push)
        echo "Pushing parquets to s3://$BUCKET ..."
        do_sync push "1_Dataset/Pre_processing/weather_data"  "1_Dataset/Pre_processing/weather_data"
        do_sync push "1_Dataset/Processed_data"               "1_Dataset/Processed_data"
        do_sync push "2_Features build/Feature_data"          "2_Features_build/Feature_data"
        do_sync push "3_Targets build/Target_data"            "3_Targets_build/Target_data"
        do_sync push "4_Features select/Feature_store"        "4_Features_select/Feature_store"
        do_sync push "4_Features select/Selected_features"    "4_Features_select/Selected_features"
        echo "Done."
        ;;
    pull)
        echo "Pulling parquets from s3://$BUCKET ..."
        do_sync pull "1_Dataset/Pre_processing/weather_data"  "1_Dataset/Pre_processing/weather_data"
        do_sync pull "1_Dataset/Processed_data"               "1_Dataset/Processed_data"
        do_sync pull "2_Features build/Feature_data"          "2_Features_build/Feature_data"
        do_sync pull "3_Targets build/Target_data"            "3_Targets_build/Target_data"
        do_sync pull "4_Features select/Feature_store"        "4_Features_select/Feature_store"
        do_sync pull "4_Features select/Selected_features"    "4_Features_select/Selected_features"
        echo "Done."
        ;;
    *)
        echo "Usage: $0 setup|push|pull"
        exit 1
        ;;
esac
