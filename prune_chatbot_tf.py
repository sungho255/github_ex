import os
# TensorFlow의 불필요한 로깅을 줄입니다.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from transformers import TFAutoModelForCausalLM, AutoTokenizer
import numpy as np

def calculate_sparsity(model):
    """
    모델의 가중치 희소성을 계산합니다.

    Args:
        model (tf.keras.Model): 희소성을 확인할 모델.

    Returns:
        float: 모델의 희소성 비율.
    """
    total_zeros = 0
    total_params = 0
    for weight in model.trainable_weights:
        total_params += tf.size(weight).numpy()
        total_zeros += tf.math.count_nonzero(weight == 0).numpy()
    return total_zeros / total_params

def main():
    """
    TensorFlow를 사용하여 챗봇 모델의 프루닝을 시연하는 메인 함수.
    """
    try:
        # 1. 사전 훈련된 챗봇 모델과 토크나이저를 로드합니다.
        model_name = "microsoft/DialoGPT-small"
        print(f"'{model_name}' 모델을 로드하는 중입니다...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # from_pt=True 플래그는 PyTorch 체크포인트를 TensorFlow 모델로 로드합니다.
        model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
        print("모델 로드 완료.")

        # 2. 프루닝 매개변수를 정의합니다.
        pruning_params = {
            'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=0.20,  # 가중치의 20%를 프루닝 목표로 설정
                begin_step=0,
                end_step=1, # 1 스텝만에 프루닝을 적용
                frequency=1
            )
        }

        # 3. 모델의 모든 Dense 레이어에 프루닝을 적용합니다.
        # Hugging Face 모델은 복잡한 구조를 가지므로, 각 Dense 레이어를 직접 래핑합니다.
        def apply_pruning_to_dense(layer):
            if isinstance(layer, tf.keras.layers.Dense):
                return tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            return layer

        # 모델의 복사본에 프루닝을 적용합니다.
        pruned_model = tf.keras.models.clone_model(
            model,
            clone_function=apply_pruning_to_dense,
        )

        print(f"모델 가중치의 {pruning_params['pruning_schedule'].target_sparsity * 100}%를 프루닝합니다.")

        # 4. 프루닝을 적용하기 위해 모델을 컴파일하고 한 스텝 학습합니다.
        pruned_model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        )

        # 프루닝을 적용하기 위한 가짜(dummy) 데이터 생성
        dummy_input = np.ones((1, 10), dtype='int32')
        dummy_output = np.ones((1, 10), dtype='int32')

        # 프루닝 콜백과 함께 모델을 한 스텝 학습하여 프루닝 마스크를 적용
        callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
        pruned_model.fit(dummy_input, dummy_output, epochs=1, callbacks=callbacks, verbose=0)
        
        # 5. 프루닝 래퍼를 제거하여 최종 모델을 얻습니다.
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
        print("프루닝 적용 및 래퍼 제거 완료.")

        # 6. 희소성을 계산하고 출력합니다.
        sparsity = calculate_sparsity(final_model)
        print(f"프루닝 후 모델 희소성: {sparsity:.2%}")

        # 7. 프루닝된 모델로 간단한 채팅을 시연합니다.
        print("\n--- 프루닝된 모델과 채팅하기 ---")
        print("종료하려면 'quit'을 입력하세요.")
        chat_history_ids = None
        while True:
            user_input = input(">> 사용자: ")
            if user_input.lower() == 'quit':
                break
            
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='tf')

            bot_input_ids = tf.concat([chat_history_ids, new_user_input_ids], axis=-1) if chat_history_ids is not None else new_user_input_ids

            chat_history_ids = final_model.generate(
                bot_input_ids, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id
            )

            response = tokenizer.decode(chat_history_ids[0][bot_input_ids.shape[-1]:], skip_special_tokens=True)
            print(f"프루닝된 DialoGPT: {response}")

    except ImportError:
        print("필요한 라이브러리를 설치해주세요: pip install tensorflow tensorflow-model-optimization transformers[tf-cpu]")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
