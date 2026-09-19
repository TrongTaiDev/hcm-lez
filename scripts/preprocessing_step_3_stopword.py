import os
import sys
import pandas as pd

                           
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'outputs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

ARTICLES_INPUT = os.path.join(INPUT_DIR, 'preprocessing_step_2_articles.parquet')
COMMENTS_INPUT = os.path.join(INPUT_DIR, 'preprocessing_step_2_comments.parquet')

ARTICLES_OUTPUT = os.path.join(OUTPUT_DIR, 'preprocessing_step_3_articles.parquet')
COMMENTS_OUTPUT = os.path.join(OUTPUT_DIR, 'preprocessing_step_3_comments.parquet')

                                 
STOPWORDS = {
    "là", "và", "của", "với", "những", "các",
    "một", "này", "đó", "đã", "đang", "sẽ",
    "thì", "mà", "hay", "khi", "nếu"
}


def remove_stopwords(text):
                                                
    if not isinstance(text, str) or not text.strip():
        return ''
    
                                                                                      
                                                                  
    words = text.split()
    
                      
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    
                         
    return ' '.join(filtered_words)


def remove_stopwords_with_progress(series, total, item_name="items"):
                                                
    results = []
    batch_size = max(1, total // 100)                                               
    
    for idx, text in enumerate(series):
        result = remove_stopwords(text)
        results.append(result)
        
                           
        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            progress = ((idx + 1) / total) * 100
            print(f"  Đã xử lý: {idx + 1}/{total} {item_name} ({progress:.1f}%)", end='\r', flush=True)
    
    print()                               
    return pd.Series(results)


def process_articles():
                             
    print("=" * 60)
    print("Xử lý articles - Stopword Removal")
    print("=" * 60)
    
                           
    if not os.path.exists(ARTICLES_INPUT):
        print(f"Không tìm thấy file {ARTICLES_INPUT}")
        return None
    
                      
    print(f"Đang đọc {ARTICLES_INPUT}...")
    df = pd.read_parquet(ARTICLES_INPUT)
    print(f"Đã đọc {len(df)} articles")
    
                                                
    print(f"Đang loại bỏ stopwords từ content_tokenized cho {len(df)} articles...")
    df['content_tokenized'] = remove_stopwords_with_progress(df['content_tokenized'], len(df), "articles")
    print("Đã loại bỏ stopwords từ content_tokenized")
    
                                      
    df['content_tokenized'] = df['content_tokenized'].astype(str)
    
                      
    print(f"Đang lưu vào {ARTICLES_OUTPUT}...")
    df.to_parquet(ARTICLES_OUTPUT, index=False, engine='pyarrow')
    print(f"Đã lưu {len(df)} articles vào {ARTICLES_OUTPUT}")
    
    return df


def process_comments():
                             
    print("\n" + "=" * 60)
    print("Xử lý comments - Stopword Removal")
    print("=" * 60)
    
                           
    if not os.path.exists(COMMENTS_INPUT):
        print(f"Không tìm thấy file {COMMENTS_INPUT}")
        return None
    
                      
    print(f"Đang đọc {COMMENTS_INPUT}...")
    df = pd.read_parquet(COMMENTS_INPUT)
    print(f"Đã đọc {len(df)} comments")
    
                                                     
    print(f"Đang loại bỏ stopwords từ comment_text_tokenized cho {len(df)} comments...")
    df['comment_text_tokenized'] = remove_stopwords_with_progress(df['comment_text_tokenized'], len(df), "comments")
    print("Đã loại bỏ stopwords từ comment_text_tokenized")
    
                                      
    df['comment_text_tokenized'] = df['comment_text_tokenized'].astype(str)
    
                      
    print(f"Đang lưu vào {COMMENTS_OUTPUT}...")
    df.to_parquet(COMMENTS_OUTPUT, index=False, engine='pyarrow')
    print(f"Đã lưu {len(df)} comments vào {COMMENTS_OUTPUT}")
    
    return df


def main():
                  
    print("\n" + "=" * 60)
    print("PREPROCESSING STEP 3: STOPWORD REMOVAL")
    print("=" * 60)
    
    print(f"\nDanh sách stopwords sẽ được loại bỏ: {', '.join(sorted(STOPWORDS))}")
    print(f"Tổng số stopwords: {len(STOPWORDS)}")
    
                    
    df_articles = process_articles()
    
                    
    df_comments = process_comments()
    
              
    print("\n" + "=" * 60)
    print("THỐNG KÊ")
    print("=" * 60)
    if df_articles is not None:
        print(f"- Số articles đã xử lý: {len(df_articles)}")
    if df_comments is not None:
        print(f"- Số comments đã xử lý: {len(df_comments)}")
    print("\nĐã tạo các file:")
    print("- outputs/preprocessing_step_3_articles.parquet")
    print("- outputs/preprocessing_step_3_comments.parquet")
    print("=" * 60)


if __name__ == '__main__':
    main()
