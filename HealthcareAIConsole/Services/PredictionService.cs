using Microsoft.ML;
using Microsoft.ML.Data;
using HealthcareAIConsole.Models;
using System.IO;
using Microsoft.ML.Trainers;

namespace HealthcareAIConsole.Services
{
    public class PredictionService
    {
        private readonly MLContext _mlContext;
        private ITransformer _model;

        public PredictionService()
        {
            _mlContext = new MLContext();
            LoadModel();
        }

        private void LoadModel()
        {
            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
            if (File.Exists(modelPath))
            {
                _model = _mlContext.Model.Load(modelPath, out var modelInputSchema);
            }
            else
            {
                TrainModel();
            }
        }

        public PatientPrediction Predict(PatientData input)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<PatientData, PatientPrediction>(_model);
            return predictionEngine.Predict(input);
        }

        public void TrainModel()
        {
            string dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "PatientData.csv");
            var data = _mlContext.Data.LoadFromTextFile<PatientData>(dataPath, hasHeader: true, separatorChar: ',');

            var dataProcessPipeline = _mlContext.Transforms.Concatenate("Features", nameof(PatientData.Age), nameof(PatientData.BloodPressure), nameof(PatientData.Cholesterol))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Outcome"))
                .Append(_mlContext.Transforms.CopyColumns("Label", "Outcome"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("Label"))
                .Append(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(new SdcaLogisticRegressionBinaryTrainer.Options { LabelColumnName = "Label", FeatureColumnName = "Features" }));

            var model = dataProcessPipeline.Fit(data);

            var modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "Model.zip");
            _mlContext.Model.Save(model, data.Schema, modelPath);
            _model = model;
        }
    }
}
