# Export to ONNX

```
optimum-cli export onnx -m naver-clova-ix/donut-base-finetuned-docvqa donut_docvqa_onnx/
```

Log

```
        -[✓] ONNX model output names match reference model (present.0.decoder.key, present.1.decoder.key, logits, present.2.decoder.key, present.0.decoder.value, present.3.decoder.key, present.3.decoder.value, present.2.decoder.value, present.1.decoder.value)
        - Validating ONNX Model output "logits":
                -[✓] (2, 1, 57532) matches (2, 1, 57532)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.0.decoder.key":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.0.decoder.value":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.1.decoder.key":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.1.decoder.value":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.2.decoder.key":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.2.decoder.value":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.3.decoder.key":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
        - Validating ONNX Model output "present.3.decoder.value":
                -[✓] (2, 16, 17, 64) matches (2, 16, 17, 64)
                -[✓] all values close (atol: 0.001)
The ONNX export succeeded with the warning: The maximum absolute difference between the output of the reference model and the ONNX exported model is not within the set tolerance 0.001:
- last_hidden_state: max diff = 0.001190185546875.
 The exported model was saved at: donut_docvqa_onnx
```

Exported ONNX files:

```
~-rw-rw-r-- 1 ubuntu ubuntu 707M Jul 17 12:58 decoder_model.onnx (not used)~
-rw-rw-r-- 1 ubuntu ubuntu 707M Jul 17 12:59 decoder_model_merged.onnx
~-rw-rw-r-- 1 ubuntu ubuntu 675M Jul 17 12:59 decoder_with_past_model.onnx (not used)~
-rw-rw-r-- 1 ubuntu ubuntu 298M Jul 17 12:58 encoder_model.onnx
```

# Inference with ONNXRuntime