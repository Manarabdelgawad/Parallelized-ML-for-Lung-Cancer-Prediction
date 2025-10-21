import logging
import sys
import os
from src.pipeline.runner import PipelineRunner
from src.utils.logger import setup_logging

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize pipeline runner
        runner = PipelineRunner()
        results = runner.run()
        
        print("pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"pipeline failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()