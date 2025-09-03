insert into strategies (name, description)
values ('OneP', 'Стратегия с двумя параметрами: multiplier и maxPercentRisk, специально разрабатывал стратегию с наименьшим количесвтом парамтеров');

-- strategy_id = 2 костыль
insert into strategy_possible_params (strategy_id, param_name, param_type)
values (2, 'multiplier', 'float'),
       (2, 'maxPercentRisk', 'float');