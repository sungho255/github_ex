import os
# TensorFlow의 불필요한 로깅을 줄입니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np

def get_dynamic_mask_tf(model, tokenizer, user_input, pruning_rate=0.2):
    """
    입력에 따라 동적으로 프루닝 마스크를 생성합니다. (TensorFlow 버전)
    """
    # 1. 활성화 값을 추출하기 위한 새로운 모델을 정의합니다.
    # 모델의 모든 Dense 레이어를 찾아 출력으로 설정합니다.
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    layer_outputs = [layer.output for layer in dense_layers]
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=layer_outputs)

    # 2. 입력을 모델에 통과시켜 활성화 값을 수집합니다.
    inputs = tokenizer(user_input, return_tensors="tf")
    activations = activation_model(inputs)

    # 3. 수집된 활성화 값을 기반으로 마스크를 생성합니다.
    masks = {}
    for i, layer in enumerate(dense_layers):
        act = activations[i]
        # 활성화 값의 L1 norm을 중요도 점수로 사용 (배치 차원과 시퀀스 차원을 합산)
        importance_scores = tf.reduce_sum(tf.abs(act), axis=[0, 1])
        
        # 프루닝할 임계값(threshold) 계산
        sorted_scores = tf.sort(importance_scores)
        threshold_index = int(pruning_rate * len(sorted_scores))
        threshold = sorted_scores[threshold_index]
        
        # 중요도가 임계값보다 낮은 뉴런을 0으로 만드는 마스크 생성
        mask = tf.cast(importance_scores > threshold, tf.float32)
        masks[layer.name] = mask
    
    return masks

def apply_dynamic_pruning_tf(model, masks):
    """ 생성된 마스크를 모델 가중치(kernel)에 일시적으로 적용합니다. """
    for layer in model.layers:
        if layer.name in masks:
            # 마스크를 가중치의 출력 차원에 맞게 브로드캐스팅하여 곱함
            mask = masks[layer.name]
            # kernel은 Dense 레이어의 가중치를 의미합니다.
            updated_kernel = layer.kernel * mask
            layer.kernel.assign(updated_kernel)

def main():
    try:
        model_name = "microsoft/DialoGPT-small"
        print(f"'{model_name}' 모델을 로드하는 중입니다...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # from_pt=True 플래그는 PyTorch 체크포인트를 TensorFlow 모델로 로드합니다.
        model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
        print("모델 로드 완료.")

        # 프루닝되지 않은 원본 모델의 가중치를 저장
        original_weights = model.get_weights()
        print("원본 모델 가중치 저장 완료.")

        print("\n--- 동적 프루닝 챗봇 데모 (TensorFlow) ---")
        print("종료하려면 'quit'을 입력하세요.")
        
        chat_history_ids = None
        while True:
            user_input = input("\n>> 사용자: ")
            if user_input.lower() == 'quit':
                break

            # 1. 매번 새로운 입력을 받으면 모델을 원본 상태로 되돌립니다.
            model.set_weights(original_weights)
            print("[알림] 모델을 원본 상태로 복원했습니다.")

            # 2. 현재 입력에 기반한 동적 프루닝 마스크를 생성합니다.
            print(f"'{user_input}' 입력에 대한 동적 마스크를 생성합니다...")
            masks = get_dynamic_mask_tf(model, tokenizer, user_input, pruning_rate=0.2)
            
            # 3. 생성된 마스크를 모델에 일시적으로 적용합니다.
            apply_dynamic_pruning_tf(model, masks)
            print("[알림] 동적 프루닝 마스크를 적용했습니다.")

            # 4. 프루닝된 모델로 응답을 생성합니다.
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='tf')
            bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids], axis=-1) if chat_history_ids is not None else new_user_input_ids
            
            chat_history_ids = model.generate(
                bot_input_ids, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            response = tokenizer.decode(chat_history_ids[0][bot_input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"동적 프루닝 챗봇 (TF): {response}")

    except ImportError:
        print("필요한 라이브러리를 설치해주세요: pip install tensorflow \"transformers[tf-cpu]\"")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
