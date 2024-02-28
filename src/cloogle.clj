(ns cloogle
  (:require [com.phronemophobic.llama :as llama]
            [clojure.java.io :as io]
            [clojure.edn :as edn]
            ;; [com.phronemophobic.clip :as clip]
            [wkok.openai-clojure.api :as openai]
            [com.phronemophobic.llama.raw-gguf :as raw-gguf]
            [datalevin.core :as d]
            [com.phronemophobic.dewey.util
             :refer [analyses-iter
                     analyses-seq]]
            [com.phronemophobic.usearch :as usearch]))


(def llama7b-path "/Users/adrian/workspace/llama.clj/models/llama-2-7b-chat.Q4_0.gguf")
(def code-llama7b-path "codellama-7b.Q4_0.gguf")
(def bge-path "/Users/adrian/workspace/llama.cpp/bge-large-en-v1.5/ggml-model-f16.gguf")
(def ctx (llama/create-context llama7b-path {:n-gpu-layers 100
                                             :embedding true}))

(def ctx2 (llama/create-context code-llama7b-path {:n-gpu-layers 100
                                                   :n-ctx 512
                                                   :embedding true}))

(def ctx3 (llama/create-context bge-path
                                {:n-gpu-layers 100
                                 :n-ctx 512
                                 :embedding true}))





(defn distinct-by
  "Returns a stateful transducer when that removes duplicates by keyfn"
  ([keyfn]
   (fn [rf]
     (let [seen (volatile! #{})]
       (fn
         ([] (rf))
         ([result] (rf result))
         ([result input]
          (let [k (keyfn input)]
            (if (contains? @seen k)
              result
              (do (vswap! seen conj k)
                  (rf result input))))))))))

(defn get-embedding
  ([ctx s]
   (get-embedding ctx s (float-array
                         (raw-gguf/llama_n_embd (:model ctx)))))
  ([ctx s ^floats arr]
   (llama/llama-update ctx s 0)
   (let [^com.sun.jna.ptr.FloatByReference
         fbr
         (raw-gguf/llama_get_embeddings ctx)
         p (.getPointer fbr)]
     (.read p 0 arr 0 (alength arr)))
   arr))

;; read each function
;; get doc strings

;; repo
;; :git/sha
;; :filename
;; :ns
;; :row
;; :name
;; :doc
;; 

(def doc-xform
  (mapcat
   (fn [repo]
     (eduction
      (filter (fn [m]
                (string? (:doc m))))
      (map (fn [vdef]
             (into (select-keys repo [:repo :git/sha])
                   (select-keys vdef [:filename :row :name :doc :ns]))))
      (-> repo
          :analysis
          :var-definitions)))))

(def var-table "var-table")
(def embedding-table "embedding-table")
(def openai-embedding-table "openai-embedding-table")

(defonce kvdb
  (doto (d/open-kv (.getCanonicalPath (io/file "kv.db")))
    (d/open-dbi var-table)
    (d/open-dbi embedding-table)
    (d/open-dbi openai-embedding-table)))

(defn index-var [idx vinfo]

  )

(def full-analysis (analyses-iter "analysis.edn.gz"))



(defn normalize-doc-string
  ([s]
   (loop [s s]
     (if (< (count (.getBytes s "utf-8"))
            500)
       s
       (recur (subs s 0 
                    (min 500
                         (dec (count s)))))))))

(defn print-progress []
  (map-indexed
   (fn [i x]
     (when (zero? (mod i 1000))
       (println (format "%,d" i)))
     x)))

(defn with-retries []
  (fn [rf]
    (let [fails (volatile! [])]
      (fn
        ([] (rf))
        ([result] (rf result))
        ([result input]
         (let [[error? new-result]
               (try
                 [false (rf result input)]
                 (catch Exception e
                   (prn e)
                   [true e]))]
           (if error?
             (do
               (vswap! fails
                       (fn [fails]
                         (let [t (.getTime (java.util.Date.))]
                           (into [t]
                                 (filter (fn [t2]
                                           (< (- t t2)
                                              (* 1000 60 5))))
                                 fails))))
               (prn "recent fail count" (count @fails))
               (if (> (count @fails) 10)
                 (throw new-result)
                 (recur result input)))
             new-result)))))))

(comment

  (def analysis
    (doall (take 200 full-analysis)))

  (def all-vars
    (eduction
     doc-xform
     full-analysis))

  





  ,)
(def api-key (:chatgpt/api-key
              (edn/read-string (slurp "secrets.edn"))))

(defn get-openai-embedding* [s]
  (let [result (openai/create-embedding
                {:model "text-embedding-3-small"
                 :input s}
                {:api-key api-key})]
    (-> result
        :data
        first
        :embedding)))

(defn fetch-openai-embeddings []
  ;; put embeddings in datalevin db
  (time
   (transduce (comp
               (map :doc)
               (print-progress)
               (with-retries)
               (map normalize-doc-string)
               (map
                (fn [doc-string]
                  (when (not (d/get-value kvdb openai-embedding-table doc-string))
                    (let [embedding (get-openai-embedding* doc-string)]
                      (d/transact-kv 
                       kvdb
                       [[:put openai-embedding-table doc-string embedding]]))))))
              (completing
               (fn [_ _]))
              nil
              all-vars)))

(def openai-index
  (usearch/init {:dimensions 1536
                 :quantization :quantization/f32}))
(comment

  (def all-vars
    (eduction
     (map (fn [i]
            (when-let [m (d/get-value kvdb var-table i)]
              (assoc m
                     ::id i))))
     (take-while some?)
     (range)))

  (def num-vars
    (transduce
     (map (constantly 1))
     +
     0
     all-vars))
  
  
  
  (def usearch-entries
    (eduction
     (comp
      (distinct-by 
       (juxt :repo :filename :name :row :git/sha))
      (map (fn [{:keys [doc]
                 ::keys [id]}]
             (let [doc-string (normalize-doc-string doc)
                   emb (d/get-value kvdb openai-embedding-table doc-string)]
               [id emb]))))
     all-vars))

  (usearch/reserve openai-index num-vars)
  (reduce
   (fn [_ [id emb]]
     (usearch/add openai-index id (float-array emb)))
   nil
   usearch-entries)

  (usearch/save openai-index "openai.usearch")

  
  
  ,)


#_(defn search [s]
  (let [emb (get-openai-embedding-memo s)
        results (usearch/search index (float-array emb) 4)]
    (into []
          (comp (map first)
                (map #(d/get-value kvdb var-table %)))
          results)
    ))

(def code-llama-index
    (usearch/init {:dimensions (raw-gguf/llama_n_embd (:model ctx2))
                   :quantization :quantization/f32}))
(usearch/load code-llama-index "code-llama.usearch")
(defn search-code-llama [s]
  (let [emb (get-embedding ctx2 s)
        results (usearch/search code-llama-index (float-array emb) 4)]
    (into []
          (comp (map first)
                (map #(d/get-value kvdb var-table %)))
          results)
    ))

(def clip-ctx (clip/create-context "/Users/adrian/workspace/clip.clj/models/CLIP-ViT-B-32-laion2B-s34B-b79K_ggml-model-f16.gguf"))
(def clip-index
  (delay
    (let [index (usearch/init {:dimensions (count
                                           ;; banana for scale
                                           (clip/text-embedding clip-ctx "banana"))
                              :quantization :quantization/f32})]
      (usearch/load index "clip.usearch")
      index)))
(defn search-clip [s]
  (let [emb (clip/text-embedding clip-ctx s)
        results (usearch/search @clip-index (float-array emb) 4)]
    (into []
          (comp (map first)
                (map #(d/get-value kvdb var-table %)))
          results)
    ))


(def get-openai-embedding-memo (memoize get-openai-embedding))

(def full-analysis (analyses-iter "analysis.edn.gz"))
(comment


  (def all-vars
    (eduction
     doc-xform
     full-analysis)

    #_(into []
          doc-xform
          analysis))

  (def index
    (usearch/init {:dimensions (raw-gguf/llama_n_embd (:model ctx3))
                   :quantization :quantization/f32}))

  (usearch/reserve index (count all-vars))
  #_(let [arr (float-array (raw-gguf/llama_n_embd (:model ctx3)))]
    (time
     (transduce (comp
                 (map :doc)
                 (map #(do (get-embedding ctx3 % arr)
                           arr))
                 (map-indexed vector))
                (completing
                 (fn [_ [k emb]]
                   (usearch/add index k emb)))
                nil
                all-vars)))


  ;; put embeddings in datalevin db
  (time
   (transduce (comp
               (map :doc)
               (map-indexed
                (fn [i doc-string]
                  [:put embedding-table i (get-embedding ctx3 doc-string)]))
               (partition-all 1000))
              (completing
               (fn [_ txns]
                 (d/transact-kv kvdb txns)))
              nil
              all-vars))

  ;; add embeddings from datalevin to usearch
  (transduce
   (comp
    (map (fn [i]
           (when-let [emb (d/get-value kvdb embedding-table i)]
             [i emb])))
    (take-while some?)
    (partition-all 1000))
   (completing
    (fn [_ chunk]
      (usearch/reserve index (+ (count index)
                                (count chunk)))
      (doseq [[i emb] chunk]
        (usearch/add index i emb))))
   nil
   (range))
  
  (usearch/save index "bge-all.usearch")

  ;; (usearch/save index "code-llama.usearch")

  ;; datalevin
  (time
   (d/transact-kv
    kvdb
    (eduction
     (map-indexed
      (fn [i m]
        [:put var-table i m]))
     all-vars)))






  ,)


(def bge-index
  (delay
    (let [index
          (usearch/init {:dimensions (raw-gguf/llama_n_embd (:model ctx3))
                         :quantization :quantization/f32})]
      (usearch/load index "bge-all.usearch")
      index)))
(defn search-bge
  ([s]
   (search-bge s 4))
  ([s n]
   (let [emb (get-embedding ctx3 s)
         results (usearch/search @bge-index (float-array emb) n)]
     (into []
           (comp (map first)
                 (map #(d/get-value kvdb var-table %)))
           results)
     )))

(comment
  (* (count index) 0.00002)
  ;; openai embeddings

  

  (def index
    (usearch/init {:dimensions 1536
                   :quantization :quantization/f32}))

  (usearch/reserve index (count all-vars))
  (time
   (transduce (comp
               (map :doc)
               (map #(get-openai-embedding-memo %))
               (map float-array)
               (map-indexed vector))
              (completing
               (fn [_ [k emb]]
                 (usearch/add index k emb)))
              nil
              all-vars))

  (usearch/save index "openai.usearch")


  ,)


(comment



  (def clip-index
    (usearch/init {:dimensions (count
                                ;; banana for scale
                                (clip/text-embedding clip-ctx "banana"))
                   :quantization :quantization/f32}))

  (usearch/reserve clip-index (count all-vars))
  
  
  (time
   (binding [clip/*num-threads* 6]
     (transduce (comp
                 (map :doc)
                 (map (fn [s]
                        (subs s 0 (min 512
                                       (count s)))))
                 (map #(clip/text-embedding clip-ctx %))
                 (map-indexed vector))
                (completing
                 (fn [_ [k emb]]
                   (usearch/add clip-index k emb)))
                nil
                all-vars)))

    (usearch/save index "clip.usearch")



  ,)


(comment

  (clojure.pprint/pprint
          (search-bge "video game"))
  ,)
