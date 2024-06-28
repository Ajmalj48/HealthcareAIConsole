using Microsoft.Extensions.Logging;
using Microsoft.ML;
using HealthcareAIConsole.Models;
using System.IO;
using Microsoft.ML.Trainers;

namespace HealthcareAIConsole.Services
{
    public class PredictionService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;
        private readonly ILogger<PredictionService> _logger;

        public PredictionService(ILogger<PredictionService> logger)
        {
            _mlContext = new MLContext();
            _logger = logger;
            LoadModel();
        }

        private void LoadModel()
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
            if (File.Exists(modelPath))
            {
                _model = _mlContext.Model.Load(modelPath, out var modelInputSchema);
                _logger.LogInformation("Model loaded from {ModelPath}", modelPath);
            }
            else
            {
                TrainModel();
            }
        }

        public PatientPrediction Predict(PatientData input)
        {
            _logger.LogInformation("Predicting outcome for patient data: Age={Age}, BloodPressure={BloodPressure}, Cholesterol={Cholesterol}", input.Age, input.BloodPressure, input.Cholesterol);
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<PatientData, PatientPrediction>(_model);
            var prediction = predictionEngine.Predict(input);
            _logger.LogInformation("Prediction: {Prediction}, Probability: {Probability}", prediction.Prediction, prediction.Probability);
            return prediction;
        }

        //public void TrainModel()
        //{
        //    string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "PatientData.csv");
        //    var data = _mlContext.Data.LoadFromTextFile<PatientData>(dataPath, hasHeader: true, separatorChar: ',');

        //    var dataProcessPipeline = _mlContext.Transforms.Concatenate("Features", nameof(PatientData.Age), nameof(PatientData.BloodPressure), nameof(PatientData.Cholesterol))
        //        .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
        //        .Append(_mlContext.Transforms.Conversion.MapValueToKey("Outcome"))
        //        .Append(_mlContext.Transforms.CopyColumns("Label", "Outcome"))
        //        .Append(_mlContext.Transforms.Conversion.MapKeyToValue("Label"))
        //        .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options { LabelColumnName = "Label", FeatureColumnName = "Features" }));

        //    var model = dataProcessPipeline.Fit(data);

        //    var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
        //    _mlContext.Model.Save(model, data.Schema, modelPath);
        //    _model = model;
        //    _logger.LogInformation("Model trained and saved to {ModelPath}", modelPath);
        //}

        public void TrainModel()
        {
            string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "PatientData.csv");
            var data = _mlContext.Data.LoadFromTextFile<PatientData>(dataPath, hasHeader: true, separatorChar: ',');

            var trainTestData = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            var dataProcessPipeline = _mlContext.Transforms.Concatenate("Features", nameof(PatientData.Age), nameof(PatientData.BloodPressure), nameof(PatientData.Cholesterol))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Outcome"))
                .Append(_mlContext.Transforms.CopyColumns("Label", "Outcome"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("Label"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options { LabelColumnName = "Label", FeatureColumnName = "Features" }));

            var model = dataProcessPipeline.Fit(trainData);

            var predictions = model.Transform(testData);
            var metrics = _mlContext.BinaryClassification.Evaluate(predictions, "Label");

            Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 Score: {metrics.F1Score:P2}");

            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
            _mlContext.Model.Save(model, trainData.Schema, modelPath);
            _model = model;
        }
    }
}
