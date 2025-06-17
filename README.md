Контейнеризированный сервис, который
1. читает поток транзакций из Kafka (`transactions`);
2. применяет препроцессинг + модель CatBoost;
3. отдаёт скор и флаг фрода в другой топик Kafka (`scores`);
4. вторым сервисом сохраняет результаты в PostgreSQL;
5. показывает интерактивный дашборд в Streamlit.

## Структура репозитория

```text
├── docker-compose.yml          
│
├── scoring_service/           
│   ├── Dockerfile
│   ├── kafka_app.py            
│   ├── models/
│   │   ├── my_catboost.cbm
│   │   └── preprocessor.pkl
│   └── src/                    
│       ├── preprocessing.py
│       └── scorer.py
│
├── saver_service/              
│   ├── Dockerfile
│   ├── saver.py                
│   └── ui.py                   
│
├── producer_demo/produce_test.py   
│
├── init_sql/schema.sql         
├── requirements.txt            
└── README.md
```
## для запуска проекта:
git clone https://github.com/dsrosemary/mlops2_homework.git
cd mlops2_homework

docker compose up --build -d
python producer_demo/produce_test.py test.csv (требует наличия test.csv, добавьте вручную)
Дашборд откроется по адресу http://localhost:8501

завершение:
docker compose down -v

