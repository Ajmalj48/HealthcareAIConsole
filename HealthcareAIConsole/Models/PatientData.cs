using Microsoft.ML.Data;

namespace HealthcareAIConsole.Models
{
    public class PatientData
    {
        [LoadColumn(0)]
        public float Age { get; set; }

        [LoadColumn(1)]
        public float BloodPressure { get; set; }

        [LoadColumn(2)]
        public float Cholesterol { get; set; }

        [LoadColumn(3)]
        public bool Outcome { get; set; }
    }

    public class PatientPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}
