import re
import os
import sys
import pandas as pd
import emoji

                                     
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

                
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTICLES_CSV = os.path.join(BASE_DIR, 'articles.csv')
COMMENTS_CSV = os.path.join(BASE_DIR, 'comments.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')


def remove_urls(text):
                             
    if not isinstance(text, str):
        return text
                        
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
                                     
    url_pattern2 = r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern2, '', text)
    return text


def remove_emojis(text):
                               
    if not isinstance(text, str):
        return text
                   
    text = emoji.replace_emoji(text, replace='')
    return text


def remove_special_chars(text):
                                                                                           
    if not isinstance(text, str):
        return text
                                                                    
                                                                           
                                                      
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
                               
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_text(text):
                                                                       
    if not isinstance(text, str):
        return text
    
                          
    text = text.lower()
    
                 
    text = remove_urls(text)
    
                   
    text = remove_emojis(text)
    
                            
    text = remove_special_chars(text)
    
    return text


def process_articles():
                                 
    print("=" * 60)
    print("Xử lý articles.csv")
    print("=" * 60)
    
                           
    if not os.path.exists(ARTICLES_CSV):
        print(f"Không tìm thấy file {ARTICLES_CSV}")
        return None
    
                  
    print(f"Đang đọc {ARTICLES_CSV}...")
    df = pd.read_csv(ARTICLES_CSV, encoding='utf-8')
    print(f"Đã đọc {len(df)} articles")
    
                           
    print("Đang chuẩn hóa cột content...")
    df['content'] = df['content'].apply(normalize_text)
    print("Đã chuẩn hóa content")
    
                                    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
                      
    output_file = os.path.join(OUTPUT_DIR, 'preprocessing_step_1_articles.parquet')
    print(f"Đang lưu vào {output_file}...")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"Đã lưu {len(df)} articles vào {output_file}")
    
    return df


def process_comments():
                                 
    print("\n" + "=" * 60)
    print("Xử lý comments.csv")
    print("=" * 60)
    
                           
    if not os.path.exists(COMMENTS_CSV):
        print(f"Không tìm thấy file {COMMENTS_CSV}")
        return None
    
                  
    print(f"Đang đọc {COMMENTS_CSV}...")
    df = pd.read_csv(COMMENTS_CSV, encoding='utf-8')
    print(f"Đã đọc {len(df)} comments")
    
                                
    print("Đang chuẩn hóa cột comment_text...")
    df['comment_text'] = df['comment_text'].apply(normalize_text)
    print("Đã chuẩn hóa comment_text")
    
                                    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
                      
    output_file = os.path.join(OUTPUT_DIR, 'preprocessing_step_1_comments.parquet')
    print(f"Đang lưu vào {output_file}...")
    df.to_parquet(output_file, index=False, engine='pyarrow')
    print(f"Đã lưu {len(df)} comments vào {output_file}")
    
    return df


def main():
                  
    print("\n" + "=" * 60)
    print("PREPROCESSING STEP 1: NORMALIZATION")
    print("=" * 60)
    
                    
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
    print("- outputs/preprocessing_step_1_articles.parquet")
    print("- outputs/preprocessing_step_1_comments.parquet")
    print("=" * 60)


if __name__ == '__main__':
    main()
