new_dict = {
  "mlflow_start":{
      "usage":                    False,
      "uri":                      "https://mlflow-somemlflowadress/",
      "experiment_name":          "log_in_test1",
      "artifact_storage":         "gs://some-lab/"
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
            "package version":    "buildpipe 0.5.a.",
            "target prediction":  "some_kpi",
            "project":            "some_project",
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
