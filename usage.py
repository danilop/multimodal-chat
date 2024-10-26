class ModelUsage:

    METRICS = {
        'inputTokens': 'Input text tokens',
        'outputTokens': 'Output text tokens',
        'totalTokens': 'Total text tokens',
        'inputTextTokenCount': 'Embedding text tokens',
        'inputMultimodalTokenCount': 'Embedding multimodal text tokens',
        'inputImageCount': 'Embedding multimodal images',
        'images': 'Generated images',
        'functionCalls': 'Function calls',
        'functionApproximateElapsedTime': 'Function approximate elapsed time',
    }
               
    def __init__(self) -> None:
        self.usage = {}
        for metric in self.METRICS.keys():
            self.usage[metric] = 0

    def __str__(self) -> str:
        output = []
        for metric, value in self.usage.items():
            output.append(f' {self.METRICS[metric]}: {value}')
        return 'Usage -' + ','.join(output)
                   
    def update(self, metric: str, value: int) -> None:
        if metric in self.METRICS.keys():
            self.usage[metric] += value
        else:
            raise ValueError(f"Invalid metric: {metric}")
