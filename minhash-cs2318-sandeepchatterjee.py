import numpy as np
# import hashlib
import logging
import re
from typing import List, Dict, Set, Tuple, Optional
import multiprocessing
import sqlite3
import json

# NLTK for stopwords
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# Hugging Face datasets library
from datasets import load_dataset

class MinHashSearcher:
    def __init__(self, 
                 num_hashes: int = 100, 
                 shingle_size: int = 4, 
                 log_level: int = logging.INFO,
                 db_path: Optional[str] = None,
                 remove_stopwords: bool = True):
        
        logging.basicConfig(
            level=log_level, 
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.num_hashes = num_hashes
        self.shingle_size = shingle_size
        self.remove_stopwords = remove_stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stopwords = set(stopwords.words('english'))
        
        self.prime = 2**61 - 1
        np.random.seed(18)  # Because my roll is CS2318
        self.hash_a = np.random.randint(1, self.prime, num_hashes)
        self.hash_b = np.random.randint(0, self.prime, num_hashes)
        
        # pre-compute and store the signatures
        self.db_path = db_path
        self._initialize_database()
        self.wiki_signatures = {}
        self.wiki_metadata = {}

    def _initialize_database(self):
        if not self.db_path:
            self.logger.info("No database path provided. Using in-memory storage.")
            return

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signatures (
                        filename TEXT PRIMARY KEY,
                        signature BLOB,
                        metadata JSON
                    )
                ''')
                conn.commit()
            self.logger.info(f"Initialized signature database at {self.db_path}")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.db_path = None

    def _normalize_text(self, text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        if self.remove_stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
        
        return text

    def _generate_shingles(self, text: str) -> Set[str]:
        """ Generate word shingles from the input text
        """
        text = self._normalize_text(text)
        words = text.split()
        
        # Handle short texts
        if len(words) < self.shingle_size:
            return set(words)

        try:
            shingles = set()
            for i in range(max(1, len(words) - self.shingle_size + 1)):
                shingle = ' '.join(words[i:i+self.shingle_size])
                shingles.add(shingle)
            return shingles
        except Exception as e:
            self.logger.error(f"Word shingle generation error: {e}")
            return set(words)

    def compute_minhash(self, shingles: Set[str]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_hashes, np.inf)
        signature = np.full(self.num_hashes, np.inf)        
        for shingle in shingles:
            shingle_hash = hash(shingle)          
            for i in range(self.num_hashes):
                hash_val = (self.hash_a[i] * shingle_hash + self.hash_b[i]) % self.prime
                signature[i] = min(signature[i], hash_val)
        return signature

    def precompute_signatures(self, dataset, num_workers: Optional[int] = None, sample_size: Optional[int] = None):
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() - 1)     
        self.logger.info(f"Precomputing signatures using {num_workers} workers...")
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))

        def compute_entry_signature(entry):
            filename = entry.get('filename', str(hash(entry['maintext'])))
            text = entry['maintext']
            shingles = self._generate_shingles(text)
            signature = self.compute_minhash(shingles)
            metadata = {
                'title': entry.get('title', ''),
                'url': entry.get('url', ''),
                'source_domain': entry.get('source_domain', ''),
                'text_length': len(text)
            }        
            return filename, signature, metadata

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(pool.map(compute_entry_signature, dataset))
        self._store_signatures(results)
        self.logger.info(f"Precomputed signatures for {len(results)} entries")

    def _store_signatures(self, results: List[Tuple[str, np.ndarray, Dict]]):
        # Clear existing storage
        self.wiki_signatures.clear()
        self.wiki_metadata.clear()

        # RAM Store
        for filename, signature, metadata in results:
            self.wiki_signatures[filename] = signature
            self.wiki_metadata[filename] = metadata

        # DB store
        if self.db_path:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    for filename, signature, metadata in results:
                        # Convert NumPy array to bytes for storage
                        signature_bytes = signature.tobytes()
                        metadata_json = json.dumps(metadata)
                        
                        cursor.execute('''
                            INSERT OR REPLACE INTO signatures 
                            (filename, signature, metadata) VALUES (?, ?, ?)
                        ''', (filename, signature_bytes, metadata_json))
                    
                    conn.commit()
            except Exception as e:
                self.logger.error(f"Database storage failed: {e}")

    def search_wikipedia(self, 
                         query: str, 
                         similarity_threshold: float = 0.7, 
                         top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        # generate query signature
        normalized_query = self._normalize_text(query)
        query_shingles = self._generate_shingles(normalized_query)
        query_signature = self.compute_minhash(query_shingles)
        
        matches = []
        
        # Search
        for filename, text_signature in self.wiki_signatures.items():
            similarity = self._jaccard_similarity(query_signature, text_signature)
            
            if similarity >= similarity_threshold:
                metadata = self.wiki_metadata.get(filename, {})
                matches.append((filename, similarity, metadata))
        
        # Sort and limit
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]

    def _jaccard_similarity(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        # edge cases
        if (np.any(sig1 == np.inf) or 
            np.any(sig2 == np.inf) or 
            len(sig1) != len(sig2)):
            return 0.0

        return np.mean(sig1 == sig2)

def main():
    try:
        # English Wikipedia dataset from Hugging Face
        dataset = load_dataset('lucadiliello/english_wikipedia', split='train')
        
        # optional signature storage
        db_path = 'wikipedia_signatures.db'
        
        # Initialize MinHash searcher
        minhash_searcher = MinHashSearcher(
            num_hashes=200, 
            shingle_size=4,  # Adjusted for word shingles
            log_level=logging.INFO,
            db_path=db_path,
            remove_stopwords=True  # Enable stopwords removal
        )
        
        # signatures sample size for testing)
        minhash_searcher.precompute_signatures(dataset, sample_size=5000)
        
        # Example queries
        queries = [
            "Machine learning algorithms",
            "Climate change impacts",
            "Artificial intelligence history"
        ]
        
        for query in queries:
            print(f"\nSearch results for query: '{query}'")
            results = minhash_searcher.search_wikipedia(
                query, 
                similarity_threshold=0.5, 
                top_k=3
            )
            
            for filename, similarity, metadata in results:
                print(f"Filename: {filename}")
                print(f"Similarity: {similarity:.2f}")
                print(f"Title: {metadata.get('title', 'N/A')}")
                print(f"URL: {metadata.get('url', 'N/A')}")
                print(f"Source Domain: {metadata.get('source_domain', 'N/A')}\n")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()