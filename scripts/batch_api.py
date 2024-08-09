from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import logging

# Configure logging
logging.basicConfig(
    filename='C:/Users/Admin/Desktop/Basudev/DSC/MLOPs/08-09/logs/logfile_API.txt',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Load artifacts
pipeline = pickle.load(open('C:/Users/Admin/Desktop/Basudev/DSC/MLOPs/08-09/artifacts/data_processing_pipeline.pkl', 'rb'))
model = pickle.load(open('C:/Users/Admin/Desktop/Basudev/DSC/MLOPs/08-09/artifacts/best_classifier.pkl', 'rb'))
label_encoder = pickle.load(open('C:/Users/Admin/Desktop/Basudev/DSC/MLOPs/08-09/artifacts/label_encoder.pkl', 'rb'))

app = FastAPI()

# Root route to handle the root path
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Batch Prediction API"}

class BatchRequest(BaseModel):
    data: dict

@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    try:
        df = pd.DataFrame.from_dict(request.data)
        logging.info(f"Received batch request with {len(df)} records.")
        
        if df.empty:
            logging.error("Received empty DataFrame.")
            raise HTTPException(status_code=400, detail="Received empty DataFrame.")
        
        # Transform the input data
        transformed_input = pipeline.transform(df)
        logging.info(f"Batch data transformed successfully")
        
        # Predict
        predictions = model.predict(transformed_input)
        decoded_predictions = label_encoder.inverse_transform(predictions)
        logging.info(f"Batch predictions completed")
        
        if not len(decoded_predictions):
            logging.error("Predictions are empty.")
            raise HTTPException(status_code=500, detail="Predictions are empty.")
        
        # Save predictions to CSV
        result_df = pd.DataFrame(decoded_predictions, columns=['Predicted Risk Category'])
        output_path = 'C:/Users/Admin/Desktop/Basudev/DSC/MLOPs/08-09/Data/output/batch_predictions.csv'
        result_df.to_csv(output_path, index=False)
        logging.info(f"Batch predictions saved to {output_path}")
        
        # Return predictions as JSON
        return result_df.to_dict(orient='records')
        
    except Exception as e:
        logging.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
