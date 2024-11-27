.timer on

.schema

SELECT
  sqlite_master.name as table_name,
  pragma_table_xinfo.name as column_name
FROM sqlite_master
JOIN pragma_table_xinfo(sqlite_master.name)
WHERE sqlite_master.sql LIKE 'CREATE VIRTUAL TABLE % USING vec0(%'
  AND pragma_table_xinfo.name LIKE '%embedding%';


.exit

insert into temp.lembed_models(model) values ('./models/mxbai-embed-xsmall-v1-f16.gguf');
select vec_length(lembed('yo'));

select :query;

select
  nyt_headlines.headline,
  vec_distance_cosine(lembed(:query), embedding) as distance
from nyt_headlines_embeddings
left join nyt_headlines on nyt_headlines.rowid = nyt_headlines_embeddings.rowid
order by distance
limit 10;


.exit


create virtual table temp.x using csv(filename="fine_food_reviews_1k.csv", header=yes);

select *
from temp.x
limit 1;
