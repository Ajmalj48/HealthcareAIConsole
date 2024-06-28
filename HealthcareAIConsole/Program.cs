using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using HealthcareAIConsole.Models;
using HealthcareAIConsole.Services;
using System;

namespace HealthcareAIConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            // Setup Dependency Injection
            var serviceProvider = new ServiceCollection()
                .AddLogging(configure => configure.AddConsole())
                .AddTransient<PredictionService>()
                .BuildServiceProvider();

            var logger = serviceProvider.GetService<ILogger<Program>>();
            logger.LogInformation("Starting application");

            var predictionService = serviceProvider.GetService<PredictionService>();

            Console.WriteLine("Enter patient data for prediction:");
            Console.Write("Age: ");
            float age = float.Parse(Console.ReadLine());

            Console.Write("Blood Pressure: ");
            float bloodPressure = float.Parse(Console.ReadLine());

            Console.Write("Cholesterol: ");
            float cholesterol = float.Parse(Console.ReadLine());

            var newPatient = new PatientData
            {
                Age = age,
                BloodPressure = bloodPressure,
                Cholesterol = cholesterol

                //Age = 45,
                //BloodPressure = 120,
                //Cholesterol = 200
            };

            var testValues = new[]
            {
                new { Age = 60, BloodPressure = 140, Cholesterol = 220 },
                new { Age = 45, BloodPressure = 130, Cholesterol = 200 },
                new { Age = 30, BloodPressure = 120, Cholesterol = 180 },
                new { Age = 50, BloodPressure = 160, Cholesterol = 240 },
                new { Age = 35, BloodPressure = 150, Cholesterol = 210 },
                new { Age = 25, BloodPressure = 110, Cholesterol = 170 },
                new { Age = 70, BloodPressure = 170, Cholesterol = 260 }
            };

            var prediction = predictionService.Predict(newPatient);
            Console.WriteLine($"Prediction: {prediction.Prediction}, Probability: {prediction.Probability}");

            logger.LogInformation("Application finished");

            // Uncomment to retrain the model
            // predictionService.TrainModel();
        }
    }
}
