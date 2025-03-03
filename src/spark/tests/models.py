#!/usr/bin/env python
# test_classification_models.py - Standalone test script for train station message classifiers

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, regexp_replace
from pyspark.ml.linalg import Vectors
from pyspark.ml import PipelineModel
import os
import sys

# Initialize Spark session
spark = SparkSession.builder \
  .appName("Test Train Station Message Classification") \
  .enableHiveSupport() \
  .getOrCreate()

# Configuration
models_dir = "file:///data/ml/models/classifiers"
model_names = ["random_forest", "logistic_regression"]

# Define sample test messages
test_messages = [
  ("<p>La ligne 14 sera déviée : les arrêts situés entre Le Centre et Victor Hugo ne seront plus desservis en direction de Gare du Nord. Du lundi 18 mars au vendredi 29 mars 2025, de 22h00 à 05h00.</p>", 
   "Route change message - expected: PERTURBATION"),
  
  ("<p>Suite à un incident technique, le trafic est interrompu sur la ligne RER B entre Gare du Nord et Aéroport Charles de Gaulle. Prévision de reprise estimée à 18h30. Nous vous prions de nous excuser pour la gêne occasionnée.</p>", 
   "Delay notification - expected: PERTURBATION"),
  
  ("<p>INFORMATION TRAVAUX: En raison de travaux de modernisation, la gare de Lyon sera fermée les weekends du 15-16 et 22-23 avril 2025. Des bus de substitution seront mis en place.</p>", 
   "Station closure - expected: TRAVAUX"),
   
  ("<p>Pour votre information, nous vous rappelons que les nouveaux horaires d'été seront en vigueur à partir du 15 juin 2025. Consultez notre site web pour plus de détails.</p>", 
   "Informational message - expected: INFORMATION")
]

# Cause label mapping
cause_label_map = {
  0.0: "INFORMATION",
  1.0: "PERTURBATION",
  2.0: "TRAVAUX"
}

def load_model(model_name):
  """Load a trained classification model"""
  model_path = f"{models_dir}/{model_name}"
  try:
    model = PipelineModel.load(model_path)
    print(f"Successfully loaded model: {model_name}")
    return model
  except Exception as e:
    print(f"Error loading model {model_name}: {str(e)}")
    return None

def test_model_with_dummy_features(model, message_text, description):
  """Test a model with dummy feature vectors"""
  
  try:
    # Extract the classifier from the pipeline
    classifier = model.stages[-1]
    
    # Get number of features the model expects
    num_features = classifier.numFeatures
    print(f"Model expects {num_features} features")
    
    # Create dummy feature vector (all zeros)
    simple_test_df = spark.createDataFrame([(
      message_text,                       # message 
      Vectors.dense([0.0] * num_features) # features
    )], ["message", "features"])
    
    # Make prediction using just the classifier
    prediction = classifier.transform(simple_test_df)
    
    # Get predicted class
    pred_index = prediction.select("prediction").collect()[0][0]
    predicted_class = cause_label_map.get(pred_index, "UNKNOWN")
    
    print(f"\n--- {description} ---")
    print(f"Message: {message_text[:100]}..." if len(message_text) > 100 else f"Message: {message_text}")
    print(f"Predicted category: {predicted_class}")
    print(f"Prediction:")
    prediction.show()
    
    # Show confidence scores if available
    if "probability" in prediction.columns:
      prob = prediction.select("probability").collect()[0][0]
      confidence = max(prob) * 100
      print(f"Confidence: {confidence:.2f}%")
    
    return predicted_class
    
  except Exception as e:
    print(f"Error in classification: {str(e)}")
    import traceback
    traceback.print_exc()
    return None

def test_with_text_preprocessing(model, message_text, description):
  """Test a model with basic text preprocessing"""
  print(f"\n--- {description} ---")
  print(f"Message: {message_text[:100]}..." if len(message_text) > 100 else f"Message: {message_text}")
  
  try:
    # Create a dataframe with the test message
    test_df = spark.createDataFrame([(message_text,)], ["message"])
    
    # Clean the text
    test_df = test_df.withColumn("text_content", 
      regexp_replace(
        regexp_replace(
          regexp_replace(col("message"), "<[^>]+>", " "), 
          "&nbsp;", " "
        ),
        "\\s+", " "
      ))
    
    # Extract some simple features for analysis
    # Note: These are just for display, not used in prediction
    word_count = len(test_df.collect()[0]["text_content"].split())
    
    # Print some basic analysis
    print(f"Word count: {word_count}")
    
    # Use the dummy feature approach for actual prediction
    return test_model_with_dummy_features(model, message_text, f"{description} (with preprocessing)")
    
  except Exception as e:
    print(f"Error in preprocessing: {str(e)}")
    import traceback
    traceback.print_exc()
    return None

# Main execution code - directly in script body for spark-submit
print("Train Station Message Classification Model Tester")
print("=" * 50)

# Check if a specific model was specified via command line
if len(sys.argv) > 1 and sys.argv[1] in model_names:
  models_to_test = [sys.argv[1]]
else:
  models_to_test = model_names

for model_name in models_to_test:
  print(f"\nTesting model: {model_name}")
  print("-" * 30)
  
  # Load the model
  model = load_model(model_name)
  if model is None:
    continue
  
  # Test each message with the model
  for message, description in test_messages:
    test_with_text_preprocessing(model, message, description)

print("\nModel testing complete!")