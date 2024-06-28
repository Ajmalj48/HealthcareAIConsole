using HealthcareAIConsole.Models;
using HealthcareAIConsole.Services;
using System;

namespace HealthcareAIConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var predictionService = new PredictionService();

            var newPatient = new PatientData
            {
                Age = 45,
                BloodPressure = 120,
                Cholesterol = 200
            };

            var prediction = predictionService.Predict(newPatient);
            Console.WriteLine($"Prediction: {prediction.Prediction}, Probability: {prediction.Probability}");

            // Uncomment to retrain the model
            // predictionService.TrainModel();
        }
    }
}
