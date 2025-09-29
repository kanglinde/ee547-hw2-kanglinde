import sys
import os
import json
import re
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler

def collect_data(src, need_data):
    data = {}
    for info in need_data:
        if info in src: data[info] = src[info]
    return data

class MyHTTPHandler(BaseHTTPRequestHandler):
    def generate_response(self, code, log_msg, data):
        content_type = "text/plain"
        response = data
        if type(data) == dict or type(data) == list:
            content_type = "application/json"
            response = json.dumps(data, indent=2)

        self.responses[code] = log_msg
        self.send_response(code)
        self.send_header('Content-type', content_type)
        self.end_headers()
        self.wfile.write(response.encode('utf-8'))

    def do_GET(self):
        # parse endpoint
        endpoint = [s for s in self.path.split('/') if s]

        # /papers
        if len(endpoint) == 1 and endpoint[0] == "papers":
            if not hasattr(self.server, "papers"):
                self.generate_response(500, f'Internal Server Error', ["500 Internal Server Error"])
            else:
                data = []
                for paper in self.server.papers:
                    paper_data = collect_data(paper, ["arxiv_id", "title", "authors", "categories"])
                    if len(paper_data) > 0:
                        data.append(paper_data)
                self.generate_response(200, f'OK ({len(data)} results)', data)

        # /papers/{arxiv_id}
        elif len(endpoint) == 2 and endpoint[0] == "papers":
            if not hasattr(self.server, "papers"):
                self.generate_response(500, f'Internal Server Error', ["500 Internal Server Error"])
            else:
                arxiv_id = endpoint[1]
                is_found = False
                for paper in self.server.papers:
                    if "arxiv_id" in paper and paper["arxiv_id"] == arxiv_id:
                        is_found = True
                        data = collect_data(paper, ["arxiv_id", "title", "authors", "abstract", "categories", "published"])
                        if "abstract_stats" in paper:
                            data["abstract_stats"] = collect_data(paper["abstract_stats"], ["total_words", "unique_words", "total_sentences"])
                        self.generate_response(200, f'OK', data)
                        break
                if not is_found:
                    self.generate_response(404, f'Not Found', f'404')
        
        # /search?q={query}
        elif len(endpoint) == 1 and re.match(r'search', endpoint[0]):
            if not hasattr(self.server, "papers"):
                self.generate_response(500, f'Internal Server Error', ["500 Internal Server Error"])
            else:
                query = endpoint[0][len("search"):]
                prefix = r'\?q='
                prefix_len = len(prefix) - 1
                if not re.match(prefix, query) or len(query[prefix_len:].strip()) == 0:
                    self.generate_response(400, f'Bad Request', f'400')
                else:
                    query = query[prefix_len:].strip()
                    data = { "query": query, "results": [] }
                    total_matches = 0
                    for paper in self.server.papers:
                        result = collect_data(paper, ["arxiv_id", "title"])
                        result["match_score"] = 0
                        result["matches_in"] = []
                        if "title" in paper:
                            count = paper["title"].lower().count(query.lower())
                            if count > 0:
                                result["match_score"] += count
                                result['matches_in'].append("title")
                        if "abstract" in paper:
                            count = paper["abstract"].lower().count(query.lower())
                            if count > 0:
                                result["match_score"] += count
                                result["matches_in"].append("abstract")
                        if result["match_score"] > 0:
                            data["results"].append(result)
                            total_matches += result["match_score"]
                    self.generate_response(200, f'OK ({total_matches} matches)', data)
        
        # /stats
        elif len(endpoint) == 1 and endpoint[0] == "stats":
            if not hasattr(self.server, "corpus"):
                self.generate_response(500, f'Internal Server Error', ["500 Internal Server Error"])
            else:
                data = {}
                corpus = self.server.corpus
                if "papers_processed" in corpus:
                    data["total_papers"] = corpus["papers_processed"]
                if "corpus_stats" in corpus:
                    if "total_words" in corpus["corpus_stats"]:
                        data["total_words"] = corpus["corpus_stats"]["total_words"]
                    if "unique_words_global" in corpus["corpus_stats"]:
                        data["unique_words"] = corpus["corpus_stats"]["unique_words_global"]
                if "top_50_words" in corpus:
                    data["top_10_words"] = []
                    for i in range(min(10, len(corpus["top_50_words"]))):
                        data["top_10_words"].append({
                            "word": corpus["top_50_words"][i]["word"],
                            "frequency": corpus["top_50_words"][i]["frequency"]
                        })
                if "category_distribution" in corpus:
                    data["category_distribution"] = corpus["category_distribution"]
                self.generate_response(200, f'OK', data)

        # invalid endpoint
        else:
            self.generate_response(404, f'Not Found', f'404')

    def log_request(self, code='-', size='-'):
        time = f'{self.log_date_time_string().replace("/", "-")}'
        cmd = f'{self.command}'
        path = f'{self.path}'
        print(f'[{time}] {cmd} {path} - {code} {self.responses[code]}', flush=True)

def main():
    # checking input
    if len(sys.argv) < 2:
        print(f'Usage: python {sys.argv[0]} <port>')
        sys.exit(1)
    port = int(sys.argv[1])
    papers_path = "sample_data/papers.json"
    corpus_path = "sample_data/corpus_analysis.json"

    # Load data
    papers = []
    print(f'Loading {papers_path}...', flush=True)
    if os.path.exists(papers_path):
        with open(papers_path, 'r') as f:
            papers = json.load(f)
    else:
        print(f'Error: missing file: {papers_path}')
        sys.exit(1)

    corpus = {}
    print(f'Loading {corpus_path}...', flush=True)
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r') as f:
            corpus = json.load(f)
    else:
        print(f'Error: missing file: {corpus_path}')
        sys.exit(1)

    # Start server
    print(f'Starting server...\n', flush=True)
    server_address = ("", port)
    server = HTTPServer(server_address, MyHTTPHandler)
    server.papers = papers
    server.corpus = corpus
    server.serve_forever()

if __name__ == "__main__":
    main()