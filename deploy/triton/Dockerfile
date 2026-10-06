FROM nvcr.io/nvidia/tritonserver:23.12-py3

COPY deploy/triton/model_repository/ /models/

CMD ["tritonserver", "--model-repository=/models", "--model-control-mode=explicit", "--load-model=fashion_model"]
