build:
	docker build -t linear_regression .

run:
	docker run -it --rm -v "$(PWD)":/app linear_regression

up:
	docker-compose up -d
down:
	docker-compose down

train:
	docker exec -it linear_regression python train.py

predict:
	docker exec -it linear_regression python predict.py

evaluate:
	docker exec -it linear_regression python bonus.py -p

plot:
	docker exec -it linear_regression python bonus.py -p_data

plot_line:
	docker exec -it linear_regression python bonus.py -p_line


