from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import os
import numpy as np
import re
from pathlib import Path
import logging
import warnings

try:
    from transformers import pipeline
except Exception:
    pipeline = None

# Suppress warnings
warnings.filterwarnings('ignore')
if pipeline is not None:
    logging.getLogger('transformers').setLevel(logging.ERROR)

app = Flask(__name__)

# Tentukan path ke folder data
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'

# Initialize POS tagging pipeline (lazy loading - akan dimuat saat pertama kali digunakan)
_pos_tagger = None

def get_pos_tagger():
    """Lazy load POS tagger pipeline (with timeout fallback)"""
    global _pos_tagger
    if _pos_tagger is None:
        if pipeline is None:
            print("Transformers not installed; using trigger word-based aspect detection")
            _pos_tagger = False
            return None

        print("Loading POS tagger model (first time only, may take a moment)...")
        try:
            # Try to load with signal timeout (Windows doesn't support signal, so we use threading approach)
            import threading
            import time
            
            load_result = {"model": None, "error": None}
            
            def load_model():
                try:
                    model = pipeline(
                        "token-classification",
                        model="w11wo/indonesian-roberta-base-posp-tagger",
                        aggregation_strategy="simple",
                        device=-1  # Use CPU for stability
                    )
                    load_result["model"] = model
                except Exception as e:
                    load_result["error"] = str(e)
            
            # Load in background thread with timeout
            loader_thread = threading.Thread(target=load_model, daemon=True)
            loader_thread.start()
            loader_thread.join(timeout=30)  # Wait max 30 seconds for model load
            
            if load_result["model"] is not None:
                _pos_tagger = load_result["model"]
                print("POS tagger model loaded successfully")
            else:
                print(f"Warning: POS tagger failed to load: {load_result['error']}")
                print("Falling back to trigger word-based aspect detection")
                _pos_tagger = False
        except Exception as e:
            print(f"Warning: Failed to load POS tagger: {e}")
            print("Falling back to trigger word-based aspect detection")
            _pos_tagger = False
    
    return _pos_tagger if _pos_tagger is not False else None

# Mapping normalisasi aspek
ASPECT_MAPPING = {
    'ruu perampasan aset': 'ruu perampasan aset',
    'uu perampasan aset': 'ruu perampasan aset',
    'uud perampasan aset': 'ruu perampasan aset',
    'undang undang perampasan aset': 'ruu perampasan aset',
    'perampasan aset': 'ruu perampasan aset',
    'uu perampasan': 'ruu perampasan aset',
    'ruu perampasan': 'ruu perampasan aset',
    'rampas aset': 'ruu perampasan aset',
    'puan': 'puan maharani',
    'puan maharani': 'puan maharani',
    'prabowo': 'prabowo',
    'prabowo subianto': 'prabowo',
    'presiden prabowo': 'prabowo',
    'presiden prabowo subianto': 'prabowo',
    'pak prabowo': 'prabowo',
    'bapak prabowo': 'prabowo',
}

def normalize_aspect(aspect):
    """Normalisasi nilai aspek berdasarkan mapping"""
    if pd.isna(aspect):
        return None
    
    aspect = str(aspect).strip().lower()
    
    # Jika kosong, "-", atau "komentar", return None
    if aspect == '' or aspect == '-' or aspect == 'komentar':
        return None
    
    # Cek mapping
    if aspect in ASPECT_MAPPING:
        return ASPECT_MAPPING[aspect]
    
    return aspect

def clean_nan_values(obj):
    """Bersihkan NaN dan infinity values untuk JSON serialization"""
    if isinstance(obj, dict):
        return {k: clean_nan_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    else:
        return obj

def load_skenario_data():
    """Load data dari hasil_skenario3.xlsx"""
    filepath = DATA_DIR / 'hasil_skenario3.xlsx'
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        print(f"Error loading skenario data: {e}")
        return pd.DataFrame()

def load_perbandingan_data():
    """Load data dari hasil_perbandingan_evaluasi.xlsx"""
    filepath = DATA_DIR / 'hasil_perbandingan_evaluasi.xlsx'
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        print(f"Error loading perbandingan data: {e}")
        return pd.DataFrame()

def load_rata_rata_data():
    """Load data dari rata_rata_evaluasi_skenario.xlsx"""
    filepath = DATA_DIR / 'rata_rata_evaluasi_skenario.xlsx'
    try:
        df = pd.read_excel(filepath)
        return df
    except Exception as e:
        print(f"Error loading rata-rata data: {e}")
        return pd.DataFrame()

# ===== FUNGSI UNTUK ANALISIS KOMENTAR TUNGGAL (Single Comment Analysis) =====
# Pipeline penelitian: input → normalisasi → slang → POS tagging → aspek POS → trigger filter → opini → sentimen

def load_aspects_list():
    """Load daftar aspek dari aspek.xlsx"""
    filepath = DATA_DIR / 'aspek.xlsx'
    try:
        df = pd.read_excel(filepath)
        if 'aspek' in df.columns:
            aspects = df['aspek'].dropna().unique()
            aspects = [str(asp).strip().lower() for asp in aspects]
            # Urutkan berdasarkan panjang frasa dari paling panjang ke pendek (greedy matching)
            aspects = sorted(aspects, key=lambda x: len(x.split()), reverse=True)
            return aspects
        return []
    except Exception as e:
        print(f"Error loading aspects: {e}")
        return []

def load_lexicon(filepath):
    """Load lexicon dari file Excel dengan kolom word dan score"""
    try:
        df = pd.read_excel(filepath)
        if 'word' in df.columns and 'score' in df.columns:
            lexicon = {}
            for _, row in df.iterrows():
                word = str(row['word']).strip().lower()
                score = float(row['score']) if not pd.isna(row['score']) else 0
                lexicon[word] = score
            return lexicon
        return {}
    except Exception as e:
        print(f"Error loading lexicon from {filepath}: {e}")
        return {}

def load_lexicon_combined():
    """Load combined lexicon dari positive + negative TERPISAH
    PENTING: Jika ada kata yang muncul di keduanya, HITUNG KEDUANYA
    Return: tuple (positive_lex, negative_lex) - untuk accumulate semua scores
    """
    positive_filepath = DATA_DIR / 'positive.xlsx'
    negative_filepath = DATA_DIR / 'negative.xlsx'
    
    positive_lex = load_lexicon(positive_filepath)
    negative_lex = load_lexicon(negative_filepath)
    
    # Return keduanya terpisah - jangan merge/override
    # Saat extract opini, ambil dari kedua lexicon dan accumulate semua scores
    return positive_lex, negative_lex

def normalisasi_teks(text):
    """
    Normalisasi teks: case folding, cleaning, dan rapikan spasi
    Tahap: lowercase → hapus karakter spesial → rapikan spasi
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    # Case folding: lowercase
    text = text.lower()
    # Hapus karakter selain huruf, angka, dan spasi
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Rapikan spasi (multiple spaces menjadi single space)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_kamus_slang():
    """Load kamus slang dari kamuskatabaku.xlsx
    Asumsikan file memiliki kolom untuk kata tidak baku dan kata baku
    Return: dict {kata_tidak_baku: kata_baku}
    """
    filepath = DATA_DIR / 'kamuskatabaku.xlsx'
    try:
        df = pd.read_excel(filepath)
        kamus = {}
        
        # Detect columns - biasanya kolom pertama adalah tidak baku, kedua adalah baku
        # atau bisa memiliki header seperti 'tidak_baku', 'baku', 'slang', 'standar', dll
        columns = df.columns.tolist()
        
        if len(columns) >= 2:
            col_tidak_baku = columns[0]
            col_baku = columns[1]
            
            for _, row in df.iterrows():
                tidak_baku = str(row[col_tidak_baku]).strip().lower() if not pd.isna(row[col_tidak_baku]) else None
                baku = str(row[col_baku]).strip().lower() if not pd.isna(row[col_baku]) else None
                
                if tidak_baku and baku:
                    kamus[tidak_baku] = baku
        
        return kamus
    except Exception as e:
        print(f"Warning: Error loading kamus slang: {e}")
        return {}

def normalisasi_slang(tokens, kamus_slang):
    """
    Normalisasi kata-kata tidak baku menjadi kata baku menggunakan kamus
    Input: list of tokens
    Output: list of normalized tokens
    """
    if not kamus_slang:
        return tokens
    
    normalized = []
    for token in tokens:
        if token in kamus_slang:
            normalized.append(kamus_slang[token])
        else:
            normalized.append(token)
    
    return normalized

def pos_tagging_komentar(text):
    """
    POS Tagging menggunakan transformers pipeline WITH TIMEOUT
    Jika timeout atau error, return empty list (fallback ke trigger word)
    Input: normalized text
    Output: list of dicts dengan keys 'word' dan 'entity' (tag POS)
    """
    tagger = get_pos_tagger()
    if not tagger:
        # POS tagger tidak tersedia, fallback ke trigger word
        return []
    
    try:
        import threading
        import time
        
        pos_results_container = {"results": None, "error": None}
        
        def do_tagging():
            try:
                # Tokenisasi sederhana untuk input
                tokens = text.split()
                if not tokens:
                    pos_results_container["results"] = []
                    return
                
                # POS tagging
                results = tagger(tokens)
                
                # Format hasil menjadi {word: tag}
                pos_results = []
                for result in results:
                    word = result['word'].lower()
                    # Hapus prefix '##' jika ada (sub-word tokens)
                    if word.startswith('##'):
                        word = word[2:]
                    
                    entity = result['entity']
                    # Standardize tag names
                    tag = entity.replace('B-', '').replace('I-', '')
                    
                    pos_results.append({
                        'word': word,
                        'tag': tag
                    })
                
                pos_results_container["results"] = pos_results
            except Exception as e:
                pos_results_container["error"] = str(e)
        
        # Run tagging in thread with 15-second timeout
        tagger_thread = threading.Thread(target=do_tagging, daemon=True)
        tagger_thread.start()
        tagger_thread.join(timeout=15)  # Max 15 seconds per request
        
        if pos_results_container["results"] is not None:
            return pos_results_container["results"]
        else:
            print(f"Warning: POS tagging timeout or error: {pos_results_container['error']}")
            return []
    
    except Exception as e:
        print(f"Error in POS tagging: {e}")
        return []

def ambil_kandidat_aspek_pos(pos_results):
    """
    Ambil kandidat aspek dari hasil POS tagging
    Kandidat aspek diambil dari tags: NOUN, PROPN, NNP, NNO
    Input: list dari dicts {'word': word, 'tag': tag}
    Output: list of candidate aspect phrases (gabung kata-kata adjacent dengan tag yang sama)
    """
    if not pos_results:
        return []
    
    # Tags yang dianggap sebagai aspek
    aspect_tags = {'NOUN', 'PROPN', 'NNP', 'NNO'}
    
    candidates = []
    current_phrase = []
    current_tag = None
    
    for result in pos_results:
        word = result['word']
        tag = result['tag']
        
        if tag in aspect_tags:
            if tag == current_tag and current_phrase:
                # Lanjutkan phrase yang sama
                current_phrase.append(word)
            else:
                # Start phrase baru
                if current_phrase:
                    candidates.append(' '.join(current_phrase))
                current_phrase = [word]
                current_tag = tag
        else:
            # Tag bukan aspek
            if current_phrase:
                candidates.append(' '.join(current_phrase))
                current_phrase = []
                current_tag = None
    
    # Jangan lupa candidate terakhir
    if current_phrase:
        candidates.append(' '.join(current_phrase))
    
    return candidates

def filter_aspek_dengan_trigger(comment_text, candidates_from_pos, aspects_list):
    """
    Filter kandidat aspek dari POS tagging dengan trigger word matching
    Langkah:
    1. Cocokkan kandidat aspek dengan daftar aspek trigger (aspek.xlsx)
    2. Gunakan word boundary regex untuk matching
    3. Urutkan aspek berdasarkan panjang (panjang ke pendek) untuk prioritas
    4. Hapus sub-aspek: jika "ruu perampasan aset" ditemukan, jangan include 
       "ruu", "perampasan", "aset", "ruu perampasan", "perampasan aset"
    
    Input:
    - comment_text: normalized text
    - candidates_from_pos: list of candidate aspects dari POS tagging
    - aspects_list: trigger word list dari aspek.xlsx
    
    Output: list of filtered aspects (unik, tanpa sub-aspek)
    """
    if not comment_text or not aspects_list:
        return []
    
    detected_aspects = []
    
    # Cocokkan dengan trigger words menggunakan word boundary regex
    for aspect in aspects_list:
        pattern = r'\b' + re.escape(aspect) + r'\b'
        if re.search(pattern, comment_text):
            detected_aspects.append(aspect)
    
    if not detected_aspects:
        return []
    
    # Hapus sub-aspek: jika ada aspek panjang, jangan include aspek pendek yang bagiannya
    final_aspects = []
    for asp in detected_aspects:
        is_substring = False
        for other_asp in detected_aspects:
            if asp != other_asp:
                asp_tokens = asp.split()
                other_tokens = other_asp.split()
                
                # Check if asp adalah subsequence dari other_asp
                if len(asp_tokens) < len(other_tokens):
                    for i in range(len(other_tokens) - len(asp_tokens) + 1):
                        if other_tokens[i:i+len(asp_tokens)] == asp_tokens:
                            is_substring = True
                            break
                
                if is_substring:
                    break
        
        if not is_substring:
            final_aspects.append(asp)
    
    # Remove duplicates sambil maintain order
    seen = set()
    result = []
    for asp in final_aspects:
        if asp not in seen:
            seen.add(asp)
            result.append(asp)
    
    return result

def tokenize_text(text):
    """Tokenize text menjadi list of words"""
    if not text:
        return []
    return text.split()

def get_tokens_for_aspect(aspect):
    """Get daftar tokens yang membentuk aspek (untuk filtering saat mencari opini)"""
    if not aspect:
        return []
    return aspect.split()

def ambil_opini_global(tokens, pos_lex, neg_lex, exclude_aspects=None):
    """
    Ambil opini secara GLOBAL dari seluruh komentar
    PENTING: Cek di KEDUA lexicon (positive dan negative) dan ACCUMULATE semua scores
    Jika kata ada di positif + negatif, KEDUANYA dihitung! Contoh: sangat (5) + sangat (-5) = 0
    
    Abaikan kata-kata yang merupakan aspek atau token aspek
    
    Input:
    - tokens: list of tokens
    - pos_lex: positive lexicon dict {word: score}
    - neg_lex: negative lexicon dict {word: score}
    - exclude_aspects: list of aspects to exclude
    
    Output: list of tuples (word, total_score) - SEMUA opini, accumulate dari kedua lexicon
    """
    if not tokens:
        return []
    
    if exclude_aspects is None:
        exclude_aspects = []
    
    # Build set of excluded words (aspek + tokens aspek)
    exclude_words = set()
    for aspect in exclude_aspects:
        exclude_words.add(aspect)
        for token in get_tokens_for_aspect(aspect):
            exclude_words.add(token)
    
    opinions = []
    seen_words = set()
    
    for token in tokens:
        if token not in exclude_words and token not in seen_words:
            total_score = 0
            found_in_lexicon = False
            
            # Cek di positive lexicon
            if pos_lex and token in pos_lex:
                total_score += pos_lex[token]
                found_in_lexicon = True
            
            # Cek di negative lexicon
            if neg_lex and token in neg_lex:
                total_score += neg_lex[token]
                found_in_lexicon = True
            
            # Jika ada di salah satu lexicon, tambahkan ke opinions
            if found_in_lexicon:
                opinions.append((token, total_score))
                seen_words.add(token)
    
    return opinions

def opini_per_aspek(tokens, aspect, pos_lex, neg_lex, all_aspects_list, window_size=3):
    """
    Ekstrak opini untuk aspek tertentu menggunakan WINDOW-BASED approach
    PENTING: Cek di KEDUA lexicon (positive dan negative) dan ACCUMULATE semua scores
    Ambil SEMUA opini dalam window, bukan hanya satu
    Jika ada lebih dari satu opini di window, SEMUA harus diambil
    
    Abaikan kata-kata yang merupakan aspek lain atau token aspek lain
    Hanya kata/frasa yang ada di lexicon yang boleh menjadi opini
    
    Input:
    - tokens: list of tokens
    - aspect: aspek yang sedang dianalisis
    - pos_lex: positive lexicon dict {word: score}
    - neg_lex: negative lexicon dict {word: score}
    - all_aspects_list: list of all detected aspects (untuk filtering aspek lain)
    - window_size: default 3 (3 kata sebelum + 3 kata sesudah)
    
    Output: list of tuples (word, total_score) - SEMUA opini dalam window, accumulate dari kedua lexicon
    """
    if not tokens or not aspect:
        return []
    
    # Find position of aspect dalam tokens
    aspect_tokens = aspect.split()
    aspect_positions = []
    
    for i in range(len(tokens) - len(aspect_tokens) + 1):
        if tokens[i:i+len(aspect_tokens)] == aspect_tokens:
            aspect_positions.append((i, i + len(aspect_tokens)))
    
    if not aspect_positions:
        return []
    
    # Build set of excluded words (aspek lain + tokens aspek lain)
    exclude_words = set()
    for other_aspect in all_aspects_list:
        if other_aspect != aspect:
            exclude_words.add(other_aspect)
            for token in get_tokens_for_aspect(other_aspect):
                exclude_words.add(token)
    
    opinions = []
    seen_words = set()
    
    # Ekstrak opini dari window sekitar aspek
    for start_idx, end_idx in aspect_positions:
        window_start = max(0, start_idx - window_size)
        window_end = min(len(tokens), end_idx + window_size)
        
        for i in range(window_start, window_end):
            # Skip words yang merupakan aspek atau token aspek
            if i < start_idx or i >= end_idx:
                token = tokens[i]
                if token not in exclude_words and token not in seen_words:
                    total_score = 0
                    found_in_lexicon = False
                    
                    # Cek di positive lexicon
                    if pos_lex and token in pos_lex:
                        total_score += pos_lex[token]
                        found_in_lexicon = True
                    
                    # Cek di negative lexicon
                    if neg_lex and token in neg_lex:
                        total_score += neg_lex[token]
                        found_in_lexicon = True
                    
                    # Jika ada di salah satu lexicon, tambahkan ke opinions
                    if found_in_lexicon:
                        opinions.append((token, total_score))
                        seen_words.add(token)
    
    return opinions

def hitung_sentimen(total_score):
    """Hitung sentimen berdasarkan total score
    total_score > 0: Positif
    total_score < 0: Negatif
    total_score = 0: Netral
    """
    if total_score > 0:
        return 'Positif'
    elif total_score < 0:
        return 'Negatif'
    else:
        return 'Netral'

def format_opini_dengan_skor(opinions):
    """
    Format opini dan skor untuk ditampilkan di UI
    Input: list of tuples (word, score)
    Output: tuple (opini_str, skor_str)
    
    Format output:
    - Opini: "bagus, buruk" (comma-separated unique words)
    - Skor: "bagus (2.0), buruk (-2.0)" (word dengan score dalam kurung)
    """
    if not opinions:
        return '-', '-'
    
    opinion_words = []
    opinion_scores_with_words = []
    
    for word, score in opinions:
        opinion_words.append(word)
        # Format: "word (score)"
        score_str = str(int(score)) if score == int(score) else str(score)
        opinion_scores_with_words.append(f"{word} ({score_str})")
    
    opini_str = ', '.join(opinion_words)
    skor_str = ', '.join(opinion_scores_with_words)
    
    return opini_str, skor_str

def analisis_komentar_tunggal(komentar):
    """
    Analisis sentimen komentar tunggal sesuai Skenario 3 penelitian
    
    Pipeline:
    1. Input komentar
    2. Case folding (lowercase)
    3. Cleaning (hapus karakter spesial)
    4. Normalisasi spasi
    5. Load kamus slang
    6. Normalisasi slang untuk tokens
    7. POS Tagging (fallback ke trigger word jika gagal)
    8. Ambil kandidat aspek dari POS tags NOUN, PROPN, NNP, NNO
    9. Cocokkan dengan trigger words dari aspek.xlsx
    10. Hapus sub-aspek
    11. Identifikasi opini:
        - 0 aspek: global opini, label "komentar"
        - 1 aspek: global opini, label aspek
        - 2+ aspek: window-based opini (window=3)
    12. Hitung skor lexicon (sum opini scores)
    13. Tentukan sentimen (total score)
    14. Bentuk label akhir ("aspek : Sentimen" atau "komentar : Sentimen")
    
    Return: dict dengan keys:
    - komentar: input original
    - hasil_analisis: list of dict {aspek, opini, skor_lexicon, total_skor, sentimen, label_akhir}
    - error: error message atau None
    """
    try:
        # Validasi input
        if not komentar or pd.isna(komentar):
            return {
                'komentar': '',
                'hasil_analisis': [],
                'error': 'Komentar tidak boleh kosong'
            }
        
        komentar_original = komentar
        
        # 1-4: Normalisasi teks (case folding, cleaning, spasi)
        normalized_text = normalisasi_teks(komentar)
        
        if not normalized_text:
            return {
                'komentar': komentar_original,
                'hasil_analisis': [],
                'error': 'Komentar kosong setelah preprocessing'
            }
        
        # Load kamus slang dan load data
        kamus_slang = load_kamus_slang()
        pos_lex, neg_lex = load_lexicon_combined()  # Load TERPISAH - positive dan negative
        aspects_list = load_aspects_list()
        
        # Tokenize awal untuk normalisasi slang
        tokens_raw = tokenize_text(normalized_text)
        
        # 5-6: Normalisasi slang
        tokens_normalized = normalisasi_slang(tokens_raw, kamus_slang)
        
        # Reconstruct text setelah normalisasi slang
        text_after_slang = ' '.join(tokens_normalized)
        
        # 7: POS Tagging (dengan fallback)
        try:
            pos_results = pos_tagging_komentar(text_after_slang)
            if pos_results:
                # 8: Ambil kandidat aspek dari POS tags
                candidates_from_pos = ambil_kandidat_aspek_pos(pos_results)
            else:
                # Fallback jika POS tagging kosong
                candidates_from_pos = []
        except Exception as e:
            print(f"Warning: POS tagging failed, using fallback: {e}")
            candidates_from_pos = []
        
        # 9-10: Filter dengan trigger words dan hapus sub-aspek
        detected_aspects = filter_aspek_dengan_trigger(text_after_slang, candidates_from_pos, aspects_list)
        
        # Re-tokenize text normalized untuk opini extraction
        tokens = tokenize_text(text_after_slang)
        
        hasil_analisis = []
        
        if not detected_aspects:
            # STRATEGI 1: TIDAK ADA ASPEK
            # Ambil opini global dari seluruh komentar, abaikan kata-kata yang adalah aspek
            # Tampilkan sebagai "komentar"
            opinions = ambil_opini_global(tokens, pos_lex, neg_lex, exclude_aspects=[])
            
            # Hitung total skor
            total_score = sum(score for _, score in opinions)
            
            # Format opini dan skor
            opini_str, skor_str = format_opini_dengan_skor(opinions)
            
            sentimen = hitung_sentimen(total_score)
            
            hasil_analisis.append({
                'aspek': 'komentar',
                'opini': opini_str,
                'skor_lexicon': skor_str,
                'total_skor': round(total_score, 2),
                'sentimen': sentimen,
                'label_akhir': f"komentar : {sentimen}"
            })
        
        elif len(detected_aspects) == 1:
            # STRATEGI 2: SATU ASPEK
            # Ambil opini global (BUKAN window), tapi abaikan aspek itu sendiri
            aspect = detected_aspects[0]
            opinions = ambil_opini_global(tokens, pos_lex, neg_lex, exclude_aspects=[aspect])
            
            # Hitung total skor
            total_score = sum(score for _, score in opinions)
            
            # Format opini dan skor
            opini_str, skor_str = format_opini_dengan_skor(opinions)
            
            sentimen = hitung_sentimen(total_score)
            
            hasil_analisis.append({
                'aspek': aspect,
                'opini': opini_str,
                'skor_lexicon': skor_str,
                'total_skor': round(total_score, 2),
                'sentimen': sentimen,
                'label_akhir': f"{aspect} : {sentimen}"
            })
        
        else:
            # STRATEGI 3: LEBIH DARI SATU ASPEK
            # Gunakan window-based extraction (window=3)
            for aspect in detected_aspects:
                opinions = opini_per_aspek(tokens, aspect, pos_lex, neg_lex, detected_aspects, window_size=3)
                
                # Hitung total skor
                total_score = sum(score for _, score in opinions)
                
                # Format opini dan skor
                opini_str, skor_str = format_opini_dengan_skor(opinions)
                
                sentimen = hitung_sentimen(total_score)
                
                hasil_analisis.append({
                    'aspek': aspect,
                    'opini': opini_str,
                    'skor_lexicon': skor_str,
                    'total_skor': round(total_score, 2),
                    'sentimen': sentimen,
                    'label_akhir': f"{aspect} : {sentimen}"
                })
        
        return {
            'komentar': komentar_original,
            'hasil_analisis': hasil_analisis,
            'error': None
        }
    
    except Exception as e:
        print(f"Error in analisis_komentar_tunggal: {e}")
        import traceback
        traceback.print_exc()
        return {
            'komentar': komentar,
            'hasil_analisis': [],
            'error': f"Terjadi kesalahan: {str(e)}"
        }

def get_dashboard_stats():
    """Hitung statistik untuk dashboard"""
    df = load_skenario_data()
    
    if df.empty:
        return {
            'total_comments': 0,
            'total_aspects': 0,
            'total_sentiments': 0,
            'dominant_sentiment': 'N/A'
        }
    
    # Total komentar (hanya yang valid/non-null)
    total_comments = int(df['Komentar'].notna().sum())
    
    # Total aspek unik (setelah normalisasi dan filter)
    normalized_aspects = [normalize_aspect(asp) for asp in df.get('Aspek', [])]
    total_aspects = len(set([asp for asp in normalized_aspects if asp is not None]))
    
    # Total label sentimen unik
    total_sentiments = 0
    if 'label_akhir' in df.columns:
        total_sentiments = df['label_akhir'].nunique()
    elif 'sentimen' in df.columns:
        total_sentiments = df['sentimen'].nunique()
    
    # Sentimen dominan
    dominant_sentiment = 'N/A'
    if 'label_akhir' in df.columns:
        sentiment_counts = df['label_akhir'].value_counts()
        if not sentiment_counts.empty:
            dominant_sentiment = sentiment_counts.index[0]
    elif 'sentimen' in df.columns:
        sentiment_counts = df['sentimen'].value_counts()
        if not sentiment_counts.empty:
            dominant_sentiment = sentiment_counts.index[0]
    
    return {
        'total_comments': total_comments,
        'total_aspects': total_aspects,
        'total_sentiments': total_sentiments,
        'dominant_sentiment': str(dominant_sentiment)
    }

def get_sentiment_distribution():
    """Hitung distribusi sentimen gabungan"""
    df = load_skenario_data()
    
    if df.empty:
        return {'Positif': 0, 'Negatif': 0, 'Netral': 0}
    
    # Gunakan kolom Sentimen untuk distribusi sentimen keseluruhan
    if 'Sentimen' in df.columns:
        sentiment_counts = df['Sentimen'].value_counts().to_dict()
    else:
        sentiment_counts = {}
    
    return {
        'Positif': sentiment_counts.get('Positif', 0),
        'Negatif': sentiment_counts.get('Negatif', 0),
        'Netral': sentiment_counts.get('Netral', 0)
    }

def get_top_aspects():
    """Hitung top 10 aspek yang paling banyak dibahas"""
    df = load_skenario_data()
    
    if df.empty or 'Aspek' not in df.columns:
        return {}
    
    # Normalisasi aspek
    normalized_aspects = [normalize_aspect(asp) for asp in df['Aspek']]
    
    # Filter out None values dan empty strings - penting untuk konsistensi dengan get_sentiment_per_aspect()
    normalized_aspects = [asp for asp in normalized_aspects if asp is not None and asp != '']
    
    aspect_counts = pd.Series(normalized_aspects).value_counts()
    
    # Ambil top 10
    top_10 = aspect_counts.head(10).to_dict()
    
    return top_10

def get_all_aspects():
    """Hitung semua aspek dengan detail sentimen (untuk tabel dengan pagination)"""
    df = load_skenario_data()
    
    if df.empty or 'Aspek' not in df.columns:
        return []
    
    # Normalisasi aspek
    df['AspekNormalized'] = df['Aspek'].apply(normalize_aspect)
    
    # Extract sentimen dari label_akhir
    def extract_sentiment_from_label(label):
        if pd.isna(label):
            return None
        label_str = str(label).strip()
        if ':' in label_str:
            parts = label_str.split(',')
            sentiments = []
            for part in parts:
                if ':' in part:
                    sentiment = part.split(':')[-1].strip()
                    sentiments.append(sentiment)
            if sentiments:
                from collections import Counter
                return Counter(sentiments).most_common(1)[0][0]
        return None
    
    df['Sentimen_Extracted'] = df['label_akhir'].apply(extract_sentiment_from_label) if 'label_akhir' in df.columns else None
    
    # Filter out None aspek
    df_filtered = df[df['AspekNormalized'].notna()].copy()
    
    # Group by aspek dan hitung sentiment
    all_aspects = []
    aspect_groups = df_filtered.groupby('AspekNormalized')
    
    for aspect_name, group_data in aspect_groups:
        if 'Sentimen_Extracted' in df.columns:
            sentiment_counts = group_data['Sentimen_Extracted'].value_counts().to_dict()
        else:
            sentiment_counts = {}
        
        all_aspects.append({
            'Aspek': aspect_name,
            'Positif': sentiment_counts.get('Positif', 0),
            'Negatif': sentiment_counts.get('Negatif', 0),
            'Netral': sentiment_counts.get('Netral', 0),
            'Total': len(group_data)
        })
    
    # Sort by Total descending
    all_aspects.sort(key=lambda x: x['Total'], reverse=True)
    
    return all_aspects

def get_sentiment_per_aspect():
    """Hitung sentimen per aspek (top 10)"""
    df = load_skenario_data()
    
    if df.empty or 'Aspek' not in df.columns or 'label_akhir' not in df.columns:
        return {}
    
    # Normalisasi aspek
    df['AspekNormalized'] = df['Aspek'].apply(normalize_aspect)
    
    # Extract sentimen dari label_akhir (format: "aspek : sentimen" atau "sentimen")
    def extract_sentiment_from_label(label):
        if pd.isna(label):
            return None
        label_str = str(label).strip()
        if ':' in label_str:
            # Format: "aspek : sentimen, aspek : sentimen, ..."
            parts = label_str.split(',')
            sentiments = []
            for part in parts:
                if ':' in part:
                    sentiment = part.split(':')[-1].strip()
                    sentiments.append(sentiment)
            # Return sentimen yang paling sering muncul
            if sentiments:
                from collections import Counter
                return Counter(sentiments).most_common(1)[0][0]
        return None
    
    df['Sentimen_Extracted'] = df['label_akhir'].apply(extract_sentiment_from_label)
    
    # Filter out None values dan empty strings
    df_filtered = df[(df['AspekNormalized'].notna()) & (df['AspekNormalized'] != '')]
    
    # Get top 10 aspects
    top_aspects = df_filtered['AspekNormalized'].value_counts().head(10).index.tolist()
    
    result = {}
    for aspect in top_aspects:
        aspect_data = df_filtered[df_filtered['AspekNormalized'] == aspect]
        sentiment_counts = aspect_data['Sentimen_Extracted'].value_counts().to_dict()
        
        result[aspect] = {
            'Positif': sentiment_counts.get('Positif', 0),
            'Negatif': sentiment_counts.get('Negatif', 0),
            'Netral': sentiment_counts.get('Netral', 0),
            'Total': len(aspect_data)
        }
    
    return result

def get_skenario_table():
    """Ambil 20 data pertama untuk tabel"""
    df = load_skenario_data()
    
    if df.empty:
        return []
    
    # Ambil 20 data pertama
    df_display = df.head(20).copy()
    
    # Tentukan kolom yang akan ditampilkan
    result = []
    for idx, row in df_display.iterrows():
        komentar = str(row.get('Komentar', '')) if not pd.isna(row.get('Komentar')) else ''
        komentar_display = komentar[:100] + '...' if len(komentar) > 100 else komentar
        
        # Get label_akhir dalam format lengkap (aspek : sentimen)
        label_akhir_raw = str(row.get('label_akhir', '')) if not pd.isna(row.get('label_akhir')) else ''
        label_akhir_display = label_akhir_raw.strip() if label_akhir_raw.strip() else '-'
        
        result.append({
            'No': int(idx) + 1,
            'Komentar': komentar_display,
            'Aspek': str(row.get('Aspek', '')) if not pd.isna(row.get('Aspek')) else '-',
            'Opini': str(row.get('Opini', '')) if not pd.isna(row.get('Opini')) else '-',
            'Skor_Lexicon': str(row.get('Skor Lexicon', '')) if not pd.isna(row.get('Skor Lexicon')) else '-',
            'Total_Skor': str(row.get('Total Skor', '')) if not pd.isna(row.get('Total Skor')) else '-',
            'Sentimen': str(row.get('Sentimen', '')) if not pd.isna(row.get('Sentimen')) else '-',
            'label_akhir': label_akhir_display
        })
    
    return result

def format_to_percentage(value):
    """Format nilai ke persentase"""
    if pd.isna(value):
        return None
    return round(float(value), 2)

# def get_evaluasi_data():
#     """Get data untuk halaman evaluasi"""
#     perbandingan_df = load_perbandingan_data()
#     rata_rata_df = load_rata_rata_data()
    
#     # Convert rata-rata ke percentage format
#     if not rata_rata_df.empty:
#         rata_rata_display = rata_rata_df.copy()
#         # Format semua kolom numeric sebagai persentase
#         for col in rata_rata_display.columns:
#             if col != 'Skenario':
#                 rata_rata_display[col] = rata_rata_display[col].apply(format_to_percentage)
#     else:
#         rata_rata_display = rata_rata_df
    
#     # Convert to records dan bersihkan NaN values
#     perbandingan_records = clean_nan_values(perbandingan_df.to_dict('records')) if not perbandingan_df.empty else []
#     rata_rata_records = clean_nan_values(rata_rata_display.to_dict('records')) if not rata_rata_display.empty else []
    
#     # Cari skenario dengan F1-score tertinggi (dari data original)
#     best_scenario = 'N/A'
#     best_f1_score = 0
    
#     if not rata_rata_df.empty:
#         # Cari kolom yang mengandung "F1" atau "micro f1"
#         f1_columns = [col for col in rata_rata_df.columns if 'f1' in col.lower() and 'micro' in col.lower()]
        
#         if not f1_columns:
#             f1_columns = [col for col in rata_rata_df.columns if 'f1' in col.lower()]
        
#         if f1_columns:
#             f1_col = f1_columns[0]
#             # Cari nilai maksimum
#             max_idx = rata_rata_df[f1_col].idxmax()
#             best_f1_score = float(rata_rata_df.loc[max_idx, f1_col])
            
#             # Cari skenario name (biasanya di kolom pertama)
#             scenario_col = rata_rata_df.columns[0]
#             best_scenario = str(rata_rata_df.loc[max_idx, scenario_col])
    
#     # Siapkan data charts (dari data original untuk akurasi)
#     chart_data = {
#         'scenarios': [],
#         'accuracy': [],
#         'precision': [],
#         'recall': [],
#         'f1score': []
#     }
    
#     if not rata_rata_df.empty:
#         for _, row in rata_rata_df.iterrows():
#             chart_data['scenarios'].append(str(row['Skenario']))
            
#             # Get metric values
#             accuracy = float(row.get('Accuracy', 0)) if 'Accuracy' in row else 0
#             precision = float(row.get('Micro Precision', 0)) if 'Micro Precision' in row else 0
#             recall = float(row.get('Micro Recall', 0)) if 'Micro Recall' in row else 0
#             f1 = float(row.get('Micro F1-score', 0)) if 'Micro F1-score' in row else 0
            
#             chart_data['accuracy'].append(accuracy)
#             chart_data['precision'].append(precision)
#             chart_data['recall'].append(recall)
#             chart_data['f1score'].append(f1)
    
#     return {
#         'perbandingan': perbandingan_records,
#         'rata_rata': rata_rata_records,
#         'best_scenario': best_scenario,
#         'best_f1_score': round(best_f1_score, 2),
#         'chart_data': chart_data
#     }

@app.route('/')
def index():
    """Dashboard utama"""
    stats = get_dashboard_stats()
    sentiment_dist = get_sentiment_distribution()
    top_aspects = get_top_aspects()
    sentiment_per_aspect = get_sentiment_per_aspect()
    skenario_table = get_skenario_table()
    all_aspects = get_all_aspects()
    
    # Clean NaN values
    stats = clean_nan_values(stats)
    sentiment_dist = clean_nan_values(sentiment_dist)
    top_aspects = clean_nan_values(top_aspects)
    sentiment_per_aspect = clean_nan_values(sentiment_per_aspect)
    skenario_table = clean_nan_values(skenario_table)
    all_aspects = clean_nan_values(all_aspects)
    
    return render_template('index.html',
                         stats=json.dumps(stats),
                         sentiment_distribution=json.dumps(sentiment_dist),
                         top_aspects=json.dumps(top_aspects),
                         sentiment_per_aspect=json.dumps(sentiment_per_aspect),
                         skenario_table=json.dumps(skenario_table),
                         all_aspects=json.dumps(all_aspects))

# @app.route('/evaluasi')
# def evaluasi():
#     """Halaman evaluasi"""
#     evaluasi_data = get_evaluasi_data()
#     evaluasi_data = clean_nan_values(evaluasi_data)
    
#     return render_template('evaluasi.html',
#                          evaluasi_data=json.dumps(evaluasi_data))

@app.route('/api/stats')
def api_stats():
    """API endpoint untuk statistik"""
    return jsonify(get_dashboard_stats())

@app.route('/api/sentiment-distribution')
def api_sentiment_distribution():
    """API endpoint untuk distribusi sentimen"""
    return jsonify(get_sentiment_distribution())

@app.route('/api/top-aspects')
def api_top_aspects():
    """API endpoint untuk top aspek"""
    return jsonify(get_top_aspects())

@app.route('/api/sentiment-per-aspect')
def api_sentiment_per_aspect():
    """API endpoint untuk sentimen per aspek"""
    return jsonify(get_sentiment_per_aspect())

@app.route('/analisis')
def analisis():
    """Halaman analisis komentar tunggal"""
    return render_template('analisis.html')

@app.route('/api/analisis', methods=['POST'])
def api_analisis():
    """API endpoint untuk analisis komentar tunggal"""
    try:
        data = request.get_json()
        komentar = data.get('komentar', '').strip() if data else ''
        
        if not komentar:
            return jsonify({
                'success': False,
                'error': 'Komentar tidak boleh kosong'
            }), 400
        
        # Analisis komentar
        hasil = analisis_komentar_tunggal(komentar)
        
        if hasil['error']:
            return jsonify({
                'success': False,
                'error': hasil['error']
            }), 400
        
        return jsonify({
            'success': True,
            'komentar': hasil['komentar'],
            'hasil_analisis': hasil['hasil_analisis']
        })
    
    except Exception as e:
        print(f"Error in api_analisis: {e}")
        return jsonify({
            'success': False,
            'error': f"Terjadi kesalahan: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)
