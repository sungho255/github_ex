import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import copy

# 전역 변수로 활성화 값을 저장할 딕셔너리
activations = {}

def get_activation(name):
    """ 특정 레이어의 출력 활성화 값을 저장하는 hook 함수 """
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def get_dynamic_mask(model, tokenizer, user_input, pruning_rate=0.2):
    """
    입력에 따라 동적으로 프루닝 마스크를 생성합니다.
    중요도 척도로는 뉴런의 출력 활성화 값의 크기를 사용합니다.
    """
    # 1. 모든 Linear 레이어에 hook을 등록하여 활성화 값을 가져옵니다.
    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            hooks.append(layer.register_forward_hook(get_activation(name)))

    # 2. 입력을 모델에 통과시켜 활성화 값을 수집합니다.
    inputs = tokenizer(user_input, return_tensors="pt")
    _ = model(**inputs)

    # 3. 등록한 hook을 모두 제거합니다.
    for hook in hooks:
        hook.remove()

    # 4. 수집된 활성화 값을 기반으로 마스크를 생성합니다.
    masks = {}
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear):
            if name in activations:
                act = activations[name]
                # 활성화 값의 L1 norm을 중요도 점수로 사용
                importance_scores = torch.abs(act).sum(dim=0)
                
                # 프루닝할 임계값(threshold) 계산
                threshold = torch.quantile(importance_scores, pruning_rate)
                
                # 중요도가 임계값보다 낮은 뉴런을 0으로 만드는 마스크 생성
                mask = (importance_scores > threshold).float()
                masks[name] = mask
    
    return masks

def apply_dynamic_pruning(model, masks):
    """ 생성된 마스크를 모델 가중치에 일시적으로 적용합니다. """
    for name, layer in model.named_modules():
        if name in masks:
            # 마스크를 가중치의 출력 차원에 맞게 브로드캐스팅하여 곱함
            layer.weight.data *= masks[name].view(1, -1)

def main():
    try:
        model_name = "microsoft/DialoGPT-small"
        print(f"'{model_name}' 모델을 로드하는 중입니다...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval() # 추론 모드로 설정
        print("모델 로드 완료.")

        # 프루닝되지 않은 원본 모델의 가중치를 저장
        original_weights = copy.deepcopy(model.state_dict())
        print("원본 모델 가중치 저장 완료.")

        print("\n--- 동적 프루닝 챗봇 데모 ---")
        print("종료하려면 'quit'을 입력하세요.")
        
        chat_history_ids = None
        while True:
            user_input = input("\n>> 사용자: ")
            if user_input.lower() == 'quit':
                break

            # 1. 매번 새로운 입력을 받으면 모델을 원본 상태로 되돌립니다.
            model.load_state_dict(original_weights)
            print("[알림] 모델을 원본 상태로 복원했습니다.")

            # 2. 현재 입력에 기반한 동적 프루닝 마스크를 생성합니다.
            print(f"'{user_input}' 입력에 대한 동적 마스크를 생성합니다...")
            masks = get_dynamic_mask(model, tokenizer, user_input, pruning_rate=0.2)
            
            # 3. 생성된 마스크를 모델에 일시적으로 적용합니다.
            apply_dynamic_pruning(model, masks)
            print("[알림] 동적 프루닝 마스크를 적용했습니다.")

            # 4. 프루닝된 모델로 응답을 생성합니다.
            new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
            bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
            
            chat_history_ids = model.generate(
                bot_input_ids, 
                max_length=1000, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True, # 동적 프루닝으로 인한 성능 저하를 보완하기 위해 샘플링 사용
                top_k=50,
                top_p=0.95
            )

            response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            print(f"동적 프루닝 챗봇: {response}")

    except ImportError:
        print("필요한 라이브러리를 설치해주세요: pip install torch transformers")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()
