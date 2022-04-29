from elasticsearch import Elasticsearch

request_body = {
    "mappings" : {
        "properties" : {
            "product_key" : { "type" : "long" },
            "full_name" : { "type" : "text" },
            "rules" : { "type" : "text" },
            "doc_vector" : { 
                "type" : "dense_vector",
                "dims": 392
              }
        }
    }
}
print("creating 'example_index' index...")
es = Elasticsearch("http://localhost:9202")
es.indices.create(index = 'word_embed', body = request_body)
