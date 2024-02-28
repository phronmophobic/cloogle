(ns cloogle.web
  (:require [com.phronemophobic.llama :as llama]
            [clojure.java.io :as io]
            [clojure.data.json :as json]
            [clojure.edn :as edn]
            [clojure.string :as str]
            [com.phronemophobic.llama.raw-gguf :as raw-gguf]
            [datalevin.core :as d]
            [com.phronemophobic.usearch :as usearch]
            [wkok.openai-clojure.api :as openai]

            ;; web stuff
            [ring.adapter.jetty9 :refer [run-jetty]]
            [ring.middleware.defaults :as ring-defaults]
            ring.util.anti-forgery
            ring.middleware.anti-forgery
            [ring.util.response :as res]
            [clojure.java.io :as io]
            bidi.ring
            [selmer.parser :as selmer]

            [hiccup2.core :as hiccup]
            hiccup.page)
  (:import java.util.concurrent.Executors))
(def bge-path "ggml-model-f16.gguf")

(defonce ctx
  (llama/create-context (.getCanonicalPath (io/file bge-path))
                          {;; :n-gpu-layers 0
                           :n-ctx 512
                           :embedding true}))

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

(def var-table "var-table")
(def embedding-table "embedding-table")

(defonce kvdb
  (delay
    (doto (d/open-kv (.getCanonicalPath (io/file "kv.db")))
      (d/open-dbi var-table)
      (d/open-dbi embedding-table))))

(def bge-index
  (delay
    (let [index
          (usearch/init {:dimensions (raw-gguf/llama_n_embd (:model ctx))
                         :quantization :quantization/f32})]
      (usearch/load index
                    (.getCanonicalPath (io/file "bge-all.usearch")))
      index)))


(defonce openai-index
  (delay
    (let [index
          (usearch/init {:dimensions 1536
                         :quantization :quantization/f32})]
      (usearch/load index
                    (.getCanonicalPath (io/file "openai.usearch")))
      index))

)

(defn search-bge*
  ([s]
   (search-bge* s 4))
  ([s n]
   (let [emb (get-embedding ctx s)
         results (usearch/search @bge-index (float-array emb) n)]
     (into []
           (comp (map first)
                 (map #(d/get-value @kvdb var-table %)))
           results))))

(defonce search-executor
  (delay
    (let [thread-factory
          (reify
            java.util.concurrent.ThreadFactory
            (newThread [this r]
              (let [thread (.newThread (Executors/defaultThreadFactory)
                                       r)]
                ;; set priority to one less than normal
                (.setPriority thread
                              (max Thread/MIN_PRIORITY
                                   (dec Thread/NORM_PRIORITY)))
                thread)))]
      (Executors/newSingleThreadExecutor thread-factory))))

(defn search-bge [s n]
  @(.submit
    ^java.util.concurrent.ExecutorService
    @search-executor
    ^java.util.concurrent.Callable
    (fn []
      (search-bge* s n))))

(def api-key (:chatgpt/api-key
              (edn/read-string (slurp "secrets.edn"))))

(defn get-openai-embedding [s]
  (let [result (openai/create-embedding
                {:model "text-embedding-3-small"
                 :input s}
                {:api-key api-key})]
    (-> result
        :data
        first
        :embedding)))

(defn search-openai [s n]
  (let [emb (get-openai-embedding s)
        results (usearch/search @openai-index (float-array emb) n)]
    (into []
          (comp (map first)
                (map #(d/get-value @kvdb var-table %)))
          results)))

(defn page [title body]
  (hiccup.page/html5
   {:lang "en"}
   (list
    [:head
     [:meta {:charset "utf-8"}]
     [:meta {:name "viewport"
             :content "width=device-width"
             :initial-scale "1"}]
     [:title title]


     [:link {:href "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css"
             :rel "stylesheet"
             :integrity "sha384-T3c6CoIi6uLrA9TneNEoa7RxnatzjcDSCmG1MXxSR1GAsXEV/Dwwykc2MPK8M2HN"
             :crossorigin "anonymous"}]
     ]
    [:body body

     ;; <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js" integrity="sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL" crossorigin="anonymous"></script>
     [:script {:src "https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"
               :integrity "sha384-C6RzsynM9kWDrMNeT87bh95OGNyZPhcTNXj1NW7RuBCsyN/o0jlpcV8Qyq46cDfL"
               :crossorigin "anonymous"}]

     ])))

(defn ->html [content]
  (str (hiccup/html content)))

(defn search-docs-handler [req]
  (let [data (with-open [is (:body req)
                         rdr (io/reader is)]
               (json/read rdr))
        {:strs [search]} data

        results
        (into []
              (map
               (fn [{:keys [repo ns name doc git/sha filename row]}]
                 {:repo (->html [:a {:href
                                          (str
                                           "https://github.com/"
                                           repo)}
                                 repo])
                  :ns (->html [:a {:href
                                        (str
                                         "https://github.com/"
                                         repo
                                         "/blob/"
                                         sha
                                         "/"
                                         filename)}
                               name])
                  :name (->html [:a {:href
                                          (str
                                           "https://github.com/"
                                           repo
                                           "/blob/"
                                           sha
                                           "/"
                                           filename
                                           "#L"
                                           row)}
                                 name])
                  :doc (->html doc)}))

              (search-openai search 50))]
    {:status 200
     :body
     (json/write-str
      {:header ["repo" "ns" "name" "doc"]
       :data results})
     :headers {"Content-Type" "application/json"}}))

(def routes
  (bidi.ring/make-handler
   ["/" {
         "favicon.ico" (fn [req]
                         (res/resource-response "favicon.ico"))
         "search-docs"
         {:post search-docs-handler}
         "doc-search.html"
         (fn [req]
           {:status 200,
            :headers {"Content-Type" "text/html"}
            :body (selmer/render-file "doc-search.html"
                                      {:anti-forgery-token
                                       ring.middleware.anti-forgery/*anti-forgery-token*})})

         true (constantly
               {:status 404
                :headers {"Content-Type" "text/html"}})}]))


(defn wrap-errors
  [handler]
  (fn [request]
    (try
      (handler request)
      (catch Exception e
        (clojure.pprint/pprint e)
        {:status 500
         :headers {"Content-Type" "text/html"}
         :body (str "There was an error. Please try again later.")}))))

(def app
  ;; routes
  (-> routes
      (wrap-errors)
      (ring-defaults/wrap-defaults ring-defaults/site-defaults)))


(comment
  (def server
    (run-jetty #'app {:port 3000
                      :host "127.0.0.1"
                      :join? false}))
  

  ,)

(defn -main [& args]
  (println "starting")
  (run-jetty #'app {:port 3000
                    :host "0.0.0.0"
                    :join? true}))
