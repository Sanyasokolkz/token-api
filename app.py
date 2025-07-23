from flask import Flask, request, jsonify
import re
import pickle
import pandas as pd
import numpy as np
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Загружаем модель при старте сервера
try:
    with open('xgboost_token_model.pkl', 'rb') as f:
        model_artifacts = pickle.load(f)
    print("✅ Модель загружена успешно!")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model_artifacts = None

def parse_token_data(text):
    """
    Парсит текстовые данные токена в структурированный формат
    """
    try:
        # Инициализируем словарь с данными токена
        token_data = {}
        
        # Парсим основные финансовые показатели
        
        # Market Cap
        mcap_match = re.search(r'MCap:\s*\$?([0-9.]+)([KMB]?)', text, re.IGNORECASE)
        if mcap_match:
            value = float(mcap_match.group(1))
            unit = mcap_match.group(2).upper()
            if unit == 'K':
                value *= 1000
            elif unit == 'M':
                value *= 1000000
            elif unit == 'B':
                value *= 1000000000
            token_data['market_cap'] = value
        else:
            token_data['market_cap'] = 0
        
        # Liquidity
        liq_match = re.search(r'Liq:\s*\$?([0-9.]+)([KMB]?)', text, re.IGNORECASE)
        if liq_match:
            value = float(liq_match.group(1))
            unit = liq_match.group(2).upper()
            if unit == 'K':
                value *= 1000
            elif unit == 'M':
                value *= 1000000
            elif unit == 'B':
                value *= 1000000000
            token_data['liquidity'] = value
        else:
            token_data['liquidity'] = 0
        
        # Volume
        vol_match = re.search(r'Volume\s+\d+min:\s*\$?([0-9.]+)([KMB]?)', text, re.IGNORECASE)
        if vol_match:
            value = float(vol_match.group(1))
            unit = vol_match.group(2).upper()
            if unit == 'K':
                value *= 1000
            elif unit == 'M':
                value *= 1000000
            elif unit == 'B':
                value *= 1000000000
            token_data['volume_1min'] = value
        else:
            token_data['volume_1min'] = 0
        
        # Last Volume
        last_vol_match = re.search(r'Last Volume:\s*\$?([0-9.]+)([KMB]?)\s+([0-9.]+)x', text, re.IGNORECASE)
        if last_vol_match:
            value = float(last_vol_match.group(1))
            unit = last_vol_match.group(2).upper()
            multiplier = float(last_vol_match.group(3))
            
            if unit == 'K':
                value *= 1000
            elif unit == 'M':
                value *= 1000000
            elif unit == 'B':
                value *= 1000000000
                
            token_data['last_volume'] = value
            token_data['last_volume_multiplier'] = multiplier
        else:
            token_data['last_volume'] = 0
            token_data['last_volume_multiplier'] = 1
        
        # Token Age (конвертируем в минуты)
        age_match = re.search(r'Token Age:\s*(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s\s*)?', text, re.IGNORECASE)
        if age_match:
            hours = int(age_match.group(1)) if age_match.group(1) else 0
            minutes = int(age_match.group(2)) if age_match.group(2) else 0
            seconds = int(age_match.group(3)) if age_match.group(3) else 0
            total_minutes = hours * 60 + minutes + seconds / 60
            token_data['token_age_numeric'] = total_minutes
        else:
            token_data['token_age_numeric'] = 0
        
        # Парсим держателей по цветам
        holders_match = re.search(r'🟢:\s*(\d+)\s*\|\s*🔵:\s*(\d+)\s*\|\s*🟡:\s*(\d+)\s*\|\s*⭕️:\s*(\d+)', text)
        if holders_match:
            token_data['green_holders'] = int(holders_match.group(1))
            token_data['blue_holders'] = int(holders_match.group(2))
            token_data['yellow_holders'] = int(holders_match.group(3))
            token_data['circle_holders'] = int(holders_match.group(4))
        else:
            token_data['green_holders'] = 0
            token_data['blue_holders'] = 0
            token_data['yellow_holders'] = 0
            token_data['circle_holders'] = 0
        
        # Парсим специальных держателей
        special_holders_match = re.search(r'🤡:\s*(\d+)\s*\|\s*🌞:\s*(\d+)\s*\|\s*🌗:\s*(\d+)\s*\|\s*🌚:\s*(\d+)', text)
        if special_holders_match:
            token_data['clown_holders'] = int(special_holders_match.group(1))
            token_data['sun_holders'] = int(special_holders_match.group(2))
            token_data['half_moon_holders'] = int(special_holders_match.group(3))
            token_data['dark_moon_holders'] = int(special_holders_match.group(4))
        else:
            token_data['clown_holders'] = 0
            token_data['sun_holders'] = 0
            token_data['half_moon_holders'] = 0
            token_data['dark_moon_holders'] = 0
        
        # Total holders
        total_holders_match = re.search(r'Total:\s*(\d+)', text)
        if total_holders_match:
            token_data['total_holders'] = int(total_holders_match.group(1))
        else:
            # Считаем как сумму всех типов держателей
            token_data['total_holders'] = (
                token_data['green_holders'] + token_data['blue_holders'] + 
                token_data['yellow_holders'] + token_data['circle_holders']
            )
        
        # Top 10 percent
        top10_match = re.search(r'Top10:\s*([0-9.]+)%', text)
        if top10_match:
            token_data['top10_percent'] = float(top10_match.group(1))
        else:
            token_data['top10_percent'] = 50.0  # default
        
        # Total percent и Total now percent
        total_match = re.search(r'Total:\s*([0-9.]+)%', text)
        if total_match:
            token_data['total_percent'] = float(total_match.group(1))
        else:
            token_data['total_percent'] = 100.0
        
        total_now_match = re.search(r'Total now:\s*([0-9.]+)%', text)
        if total_now_match:
            token_data['total_now_percent'] = float(total_now_match.group(1))
        else:
            token_data['total_now_percent'] = 100.0
        
        # Insiders
        insiders_match = re.search(r'Insiders:\s*(\d+)\s+hold\s+([0-9.]+)%', text)
        if insiders_match:
            token_data['insiders_count'] = int(insiders_match.group(1))
            token_data['insiders_percent'] = float(insiders_match.group(2))
        else:
            token_data['insiders_count'] = 0
            token_data['insiders_percent'] = 0.0
        
        # Snipers
        snipers_match = re.search(r'Snipers:\s*(\d+)', text)
        if snipers_match:
            token_data['snipers_count'] = int(snipers_match.group(1))
        else:
            token_data['snipers_count'] = 0
        
        # Bundle
        bundle_total_match = re.search(r'Bundle:.*?Total:\s*(\d+)', text, re.DOTALL)
        if bundle_total_match:
            token_data['bundle_total'] = int(bundle_total_match.group(1))
        else:
            token_data['bundle_total'] = 0
        
        bundle_supply_match = re.search(r'Supply:\s*([0-9.]+)%', text)
        if bundle_supply_match:
            token_data['bundle_supply_percent'] = float(bundle_supply_match.group(1))
        else:
            token_data['bundle_supply_percent'] = 0.0
        
        # Dev holds
        dev_holds_match = re.search(r'Dev holds\s+([0-9.]+)%', text)
        if dev_holds_match:
            token_data['dev_holds_percent'] = float(dev_holds_match.group(1))
        else:
            token_data['dev_holds_percent'] = 0.0
        
        # Создаем дополнительные признаки (как в модели)
        token_data['volume_to_liquidity'] = float(
            np.log1p(token_data['volume_1min']) / np.log1p(token_data['liquidity'] + 1)
            if token_data['liquidity'] > 0 else 0
        )
        
        # Логарифмические признаки
        token_data['log_market_cap'] = float(np.log1p(token_data['market_cap']))
        token_data['log_liquidity'] = float(np.log1p(token_data['liquidity']))
        token_data['log_volume_1min'] = float(np.log1p(token_data['volume_1min']))
        token_data['log_last_volume'] = float(np.log1p(token_data['last_volume']))
        
        # Концентрация держателей
        token_data['holder_concentration'] = int(
            token_data['green_holders'] + token_data['blue_holders'] + 
            token_data['yellow_holders'] + token_data['circle_holders']
        )
        
        # Риск-индикаторы
        token_data['total_risk_percent'] = float(
            token_data['dev_holds_percent'] + token_data['insiders_percent']
        )
        
        # Конвертируем все значения в JSON-сериализуемые типы
        return convert_to_json_serializable(token_data)
        
    except Exception as e:
        logging.error(f"Ошибка парсинга: {e}")
        return {}

def convert_to_json_serializable(obj):
    """Конвертирует numpy типы в стандартные Python типы для JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

def predict_token_success(token_data):
    """
    Предсказывает успешность токена (с исправлением дрифта данных)
    """
    if model_artifacts is None:
        return {'error': 'Модель не загружена'}
    
    try:
        # Создаем DataFrame
        df_new = pd.DataFrame([token_data])
        
        # Добавляем недостающие столбцы
        for col in model_artifacts['feature_names']:
            if col not in df_new.columns:
                df_new[col] = 0
        
        # Берем только нужные признаки в правильном порядке
        df_new = df_new[model_artifacts['feature_names']]
        
        # Применяем импутер
        df_imputed = pd.DataFrame(
            model_artifacts['imputer'].transform(df_new), 
            columns=model_artifacts['feature_names']
        )
        
        # Получаем исходные предсказания модели
        raw_prediction = int(model_artifacts['model'].predict(df_imputed)[0])
        raw_probability = float(model_artifacts['model'].predict_proba(df_imputed)[0, 1])
        
        # 🔄 ИСПРАВЛЕНИЕ ДРИФТА: Инвертируем вероятность
        # Поскольку модель работает наоборот из-за дрифта данных
        probability = 1.0 - raw_probability
        prediction = int(probability >= 0.5)
        
        # Определяем уровень уверенности
        confidence_score = abs(probability - 0.5) * 2
        if confidence_score > 0.8:
            confidence_level = "Очень высокая"
        elif confidence_score > 0.6:
            confidence_level = "Высокая"
        elif confidence_score > 0.4:
            confidence_level = "Средняя"
        else:
            confidence_level = "Низкая"
        
        # Определяем рекомендацию
        if probability >= 0.7:
            recommendation = "ПОКУПАТЬ"
            risk_level = "Низкий"
        elif probability >= 0.5:
            recommendation = "ИЗУЧИТЬ"
            risk_level = "Средний"
        else:
            recommendation = "ПРОПУСТИТЬ"
            risk_level = "Высокий"
        
        result = {
            'prediction': 'ДА' if prediction == 1 else 'НЕТ',
            'probability': round(float(probability), 4),
            'probability_percent': f"{probability*100:.1f}%",
            'confidence_level': confidence_level,
            'confidence_score': round(float(confidence_score), 4),
            'recommendation': recommendation,
            'risk_level': risk_level,
            'threshold_conservative': 'ДА' if probability >= 0.7 else 'НЕТ',
            'threshold_optimal': 'ДА' if probability >= 0.5 else 'НЕТ',
            'threshold_aggressive': 'ДА' if probability >= 0.3 else 'НЕТ',
            
            # Информация о коррекции для отладки
            'drift_correction': {
                'raw_probability': round(raw_probability, 4),
                'corrected_probability': round(probability, 4),
                'inversion_applied': True,
                'note': 'Применена коррекция дрифта данных'
            }
        }
        
        # Конвертируем все в JSON-сериализуемые типы
        return convert_to_json_serializable(result)
        
    except Exception as e:
        logging.error(f"Ошибка предсказания: {e}")
        return {'error': f'Ошибка предсказания: {str(e)}'}

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка работоспособности API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_artifacts is not None,
        'version': '2.0',
        'drift_correction': 'enabled'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной эндпоинт для предсказания"""
    try:
        # Получаем данные из запроса
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Необходимо передать поле "text" с данными токена'}), 400
        
        # Парсим текст
        token_data = parse_token_data(data['text'])
        
        if not token_data:
            return jsonify({'error': 'Не удалось распарсить данные токена'}), 400
        
        # Получаем предсказание
        result = predict_token_success(token_data)
        
        # Добавляем распарсенные данные для отладки
        if data.get('include_parsed_data', False):
            result['parsed_data'] = convert_to_json_serializable(token_data)
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Ошибка API: {e}")
        return jsonify({'error': f'Внутренняя ошибка сервера: {str(e)}'}), 500

@app.route('/test', methods=['GET'])
def test():
    """Тестовый эндпоинт с примером данных"""
    test_text = """🍀 Maybe Scalp it? ✯✯✯✯✯

🎲 $lazycoin | does nothing

8erVzAoR4yD22fmQYB1kUGn9uLQuDaLZjd85wnhWbonk | 𝕏s
└ Launchpad: LetsBonk.fun

⏳Token Age: 58m 32s
 ├ MCap: $88K
 └ Liq: $40K
 └ Volume 5min: $73k 🟡
 └ Last Volume: $4k 14.9x 🟣
> First 70 holders:


🌚🌚🌚🌚🌚⭕️⭕️⭕️⭕️⭕️
⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️
⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️
⭕️⭕️⭕️⭕️⭕️⭕️🟢⭕️⭕️⭕️
⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️⭕️
🟡🟡⭕️⭕️⭕️⭕️⭕️⭕️🔵🟡
⭕️⭕️⭕️⭕️🟡⭕️⭕️⭕️⭕️⭕️

├ 🟢: 1 | 🔵: 1 | 🟡: 4 | ⭕️: 59
├ 🤡: 0 | 🌞: 0 | 🌗: 0 | 🌚: 5
├ Total: 87.5%
└ Total now: 1.1%
> Holders:
 ├ Top10: 23.7% 🟢| Total: 406
 ├ Insiders: 0 hold 0% 🟢
 └ Snipers: 5
> Bundle:
 ├ Total: 0
 └ Supply: 0% 🟢
> Dev:
 ├ Migrations: 2/319 (0.6%)
 └ Dev holds 0% 🟢"""
    
    token_data = parse_token_data(test_text)
    result = predict_token_success(token_data)
    result['parsed_data'] = convert_to_json_serializable(token_data)
    
    return jsonify(result)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Railway передает порт через переменную
    print(f"🚀 Запускаем Token Prediction API v2.0 на порту {port}")
    print(f"📊 Коррекция дрифта данных: ВКЛЮЧЕНА")
    print(f"🧪 Тест: http://localhost:{port}/test")
    print(f"❤️  Статус: http://localhost:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=False)
