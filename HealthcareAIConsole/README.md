# Healthcare AI Console Application

## Overview

This console application uses machine learning to predict health outcomes based on patient data. It is built using C# and .NET Core.

## Features

- **Prediction Service**: Predicts health outcomes (true/false) based on patient data input.
- **Model Training**: Automatically retrains the model when the application starts if no trained model is found.
- **Customizable Threshold**: Allows adjustment of prediction sensitivity by modifying the probability threshold.
- **Console Interface**: Provides a simple command-line interface for entering patient data and viewing predictions.

## Getting Started

### Prerequisites

- [.NET Core SDK](https://dotnet.microsoft.com/download) installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your/repository.git
   cd HealthcareAIConsole


### Test values
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

