import logging
from src.pipeline.runner import PipelineRunner
from src.utils.logger import setup_logging

def main():
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Run pipeline
        runner = PipelineRunner()
        results = runner.run()
        
        # Print results
        print("\n=== Pipeline Results ===")
        for result in results:
            name, acc, prec, rec, f1, model = result
            print(f"{name}: Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()