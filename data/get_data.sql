WITH 
    -- Получаем все уникальные дни
    all_days AS (
        SELECT DISTINCT formatDateTime(toDate(timestamp), '%Y-%m-%d') AS day
        FROM default.demand_orders
    ),
    
    -- Получаем все уникальные товары
    all_skus AS (
        SELECT DISTINCT sku_id, sku, price
        FROM default.demand_orders
    ),
    
    -- Рассчитываем суммарное количество продаж товаров за каждый день
    daily_sales AS (
        SELECT
            formatDateTime(toDate(timestamp), '%Y-%m-%d') AS day,
            sku_id,
            SUM(qty) AS qty
        FROM default.demand_orders
        GROUP BY day, sku_id
    )
    
-- Создаем комбинацию всех товаров и всех дней
SELECT
    all_days.day AS day,
    all_skus.sku_id AS sku_id,
    all_skus.sku AS sku,
    all_skus.price AS price,
    COALESCE(daily_sales.qty, 0) AS qty
FROM
    all_days
    CROSS JOIN all_skus
    LEFT JOIN daily_sales
        ON all_days.day = daily_sales.day
        AND all_skus.sku_id = daily_sales.sku_id
ORDER BY all_skus.sku_id ASC, all_days.day ASC;
