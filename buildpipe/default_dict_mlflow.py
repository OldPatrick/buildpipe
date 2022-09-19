new_dict = {
  "mlflow_start":{
      "usage":                    False,
      "uri":                      "https://mlflow-internal.onair.lab.da-service.io/",
      "experiment_name":          "log_in_test1",
      "artifact_storage":         "gs://mgr-onair-lab-2c1k-mlflow-lab/"
  },
  "mlflow_run": {
    "meta":{
        "run_name":         ""
    },
    "trend_season":{
        "run_name":         ""
    },
    "residual":{
        "run_name":         ""
    },
    "baseline":{
        "run_name":         ""
    },
    "mlflow_tags": {
        "fixed_parent_run": {
            "package version":    "buildpipe 0.1.a.",
            "target prediction":  "market_share",
            "project":            "onAIr",
            "task":               "experimental_task",
            "model_type":         "regression"
        },
        "automatic_parent_run": {
            "features":             "",
            "subestimator":         "",
            "window_class":         "",
            "window_train":         "",
            "window_test":          "",
            "window_sliding":       ""
      }


    }
  },
  "model": {

    "predict_changes": {
      "predict_proba": False
    }
  }
}
