                      
                       
\
\
\
   

import os
import sys
import re
import pandas as pd
import py_vncorenlp

                           
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'outputs')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

ARTICLES_INPUT = os.path.join(INPUT_DIR, 'preprocessing_step_1_articles.parquet')
COMMENTS_INPUT = os.path.join(INPUT_DIR, 'preprocessing_step_1_comments.parquet')

ARTICLES_OUTPUT = os.path.join(OUTPUT_DIR, 'preprocessing_step_2_articles.parquet')
COMMENTS_OUTPUT = os.path.join(OUTPUT_DIR, 'preprocessing_step_2_comments.parquet')

VNCORENLP_SAVE_DIR = os.path.join(BASE_DIR, 'VnCoreNLP')

                                                        
                                                     
COMPOUND_WORDS = [
    "xe xăng",
    "xe điện",
    "xe máy",
    "xe buýt",
    "xe ôm",
    "xe công nghệ",
    "xe giao hàng",
    "xe taxi",
    "xe khách",
    "xe tải",
    "tp hcm",
    "tp hà nội",
    "hà nội",
    "thành phố",
    "ô nhiễm",
    "khí thải",
    "môi trường",
    "giao thông",
    "phương tiện",
    "năng lượng",
    "trạm sạc",
    "xe đạp",
    "xe công cộng",
    "xe cá nhân",
    "xe xăng dầu",
    "xe chạy xăng",
    "xe chạy điện",
    "chuyển đổi",
    "kiểm soát",
    "hạn chế",
    "cấm xe",
    "vùng lez",
    "phát thải",
    "bụi mịn",
    "không khí",
    "bầu không khí",
    "ô tô",
    "xe hơi",
    "xe cộ",
    "xe cũ",
    "xe mới",
    "xe cũ",
    "xe xăng cũ",
    "xe điện mới",
    "trung tâm",
    "khu vực",
    "vành đai",
    "giờ cao điểm",
    "giờ thấp điểm",
    "dịch vụ",
    "gọi xe",
    "đặt xe",
    "tài xế",
    "người dân",
    "người lao động",
    "hộ gia đình",
    "hộ nghèo",
    "hộ cận nghèo",
    "thu nhập",
    "chi phí",
    "tiết kiệm",
    "hỗ trợ",
    "chính sách",
    "quy định",
    "quy chuẩn",
    "tiêu chuẩn",
    "lộ trình",
    "kế hoạch",
    "đề án",
    "dự án",
    "chương trình",
    "thực hiện",
    "triển khai",
    "áp dụng",
    "ban hành",
    "phê duyệt",
    "đồng thuận",
    "phản đối",
    "ủng hộ",
    "không ủng hộ",
    "đồng ý",
    "không đồng ý",
    "tán thành",
    "phản đối",
    "đồng tình",
    "không đồng tình",
    "hài lòng",
    "không hài lòng",
    "thích",
    "không thích",
    "yêu thích",
    "ghét",
    "tốt",
    "xấu",
    "tích cực",
    "tiêu cực",
    "lợi ích",
    "bất lợi",
    "thuận lợi",
    "khó khăn",
    "thách thức",
    "cơ hội",
    "rủi ro",
    "thành công",
    "thất bại",
    "hiệu quả",
    "không hiệu quả",
    "thành công",
    "thất bại",
    "tốt",
    "xấu",
    "tích cực",
    "tiêu cực",
    "lợi ích",
    "bất lợi",
    "thuận lợi",
    "khó khăn",
    "thách thức",
    "cơ hội",
    "rủi ro",
    "thành công",
    "thất bại",
    "hiệu quả",
    "không hiệu quả",
]


def fix_compound_words(text):
                                                       
    if not isinstance(text, str) or not text.strip():
        return text
    
                                                            
    sorted_compounds = sorted(COMPOUND_WORDS, key=len, reverse=True)
    
    for compound in sorted_compounds:
                                                                               
                                                                                
        words = compound.split()
        if len(words) > 1:
                                                                                      
            pattern = r'\b' + r'\s+'.join([re.escape(w) for w in words]) + r'\b'
            replacement = '_'.join(words)
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    return text


def word_segment_text(annotator, text):
                                                                     
    if not isinstance(text, str) or not text.strip():
        return ''
    
    try:
                           
        segmented = annotator.word_segment(text)
                                   
        if segmented:
                                              
            if isinstance(segmented, list):
                if len(segmented) > 0:
                                                   
                    if isinstance(segmented[0], list):
                        result = ' '.join([' '.join(sentence) for sentence in segmented])
                    else:
                        result = ' '.join(segmented)
                else:
                    result = ''
            else:
                                                    
                result = str(segmented)
            
                                                                 
            result = fix_compound_words(result)
            return result
        return ''
    except Exception as e:
        print(f"Lỗi khi word segment: {e}")
        return text                              


def word_segment_with_progress(annotator, series, total, item_name="items"):
                                           
    results = []
    batch_size = max(1, total // 100)                                               
    
    for idx, text in enumerate(series):
        result = word_segment_text(annotator, text)
        results.append(result)
        
                           
        if (idx + 1) % batch_size == 0 or (idx + 1) == total:
            progress = ((idx + 1) / total) * 100
            print(f"  Đã xử lý: {idx + 1}/{total} {item_name} ({progress:.1f}%)", end='\r', flush=True)
    
    print()                               
    return pd.Series(results)


def process_articles(annotator):
                             
    print("=" * 60)
    print("Xử lý articles - Tokenization")
    print("=" * 60)
    
                           
    if not os.path.exists(ARTICLES_INPUT):
        print(f"Không tìm thấy file {ARTICLES_INPUT}")
        return None
    
                      
    print(f"Đang đọc {ARTICLES_INPUT}...")
    df = pd.read_parquet(ARTICLES_INPUT)
    print(f"Đã đọc {len(df)} articles")
    
                              
    print(f"Đang word segment cột content cho {len(df)} articles...")
    df['content_tokenized'] = word_segment_with_progress(annotator, df['content'], len(df), "articles")
    print("Đã word segment content")
    
                                      
    df['content_tokenized'] = df['content_tokenized'].astype(str)
    
                      
    print(f"Đang lưu vào {ARTICLES_OUTPUT}...")
    df.to_parquet(ARTICLES_OUTPUT, index=False, engine='pyarrow')
    print(f"Đã lưu {len(df)} articles vào {ARTICLES_OUTPUT}")
    
    return df


def process_comments(annotator):
                             
    print("\n" + "=" * 60)
    print("Xử lý comments - Tokenization")
    print("=" * 60)
    
                           
    if not os.path.exists(COMMENTS_INPUT):
        print(f"Không tìm thấy file {COMMENTS_INPUT}")
        return None
    
                      
    print(f"Đang đọc {COMMENTS_INPUT}...")
    df = pd.read_parquet(COMMENTS_INPUT)
    print(f"Đã đọc {len(df)} comments")
    
                                   
    print(f"Đang word segment cột comment_text cho {len(df)} comments...")
    df['comment_text_tokenized'] = word_segment_with_progress(annotator, df['comment_text'], len(df), "comments")
    print("Đã word segment comment_text")
    
                                      
    df['comment_text_tokenized'] = df['comment_text_tokenized'].astype(str)
    
                      
    print(f"Đang lưu vào {COMMENTS_OUTPUT}...")
    df.to_parquet(COMMENTS_OUTPUT, index=False, engine='pyarrow')
    print(f"Đã lưu {len(df)} comments vào {COMMENTS_OUTPUT}")
    
    return df


def main():
                  
    print("\n" + "=" * 60)
    print("PREPROCESSING STEP 2: TOKENIZATION")
    print("=" * 60)
    
                                           
    os.makedirs(VNCORENLP_SAVE_DIR, exist_ok=True)
    
                                     
    print(f"\nĐang kiểm tra/tải model VnCoreNLP (save_dir: {VNCORENLP_SAVE_DIR})...")
    print("  (Quá trình này có thể mất vài phút nếu chưa có model...)")
    try:
        py_vncorenlp.download_model(save_dir=VNCORENLP_SAVE_DIR)
        print("Model VnCoreNLP đã sẵn sàng!")
    except Exception as e:
        print(f"Lưu ý khi tải model: {e}")
        print("  (Có thể model đã tồn tại hoặc đang được tải...)")
    
                                  
    print(f"\nĐang khởi tạo VnCoreNLP...")
    print("  (Quá trình này có thể mất vài giây...)")
    try:
                                           
        rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"], 
            save_dir=VNCORENLP_SAVE_DIR
        )
        print("Đã khởi tạo VnCoreNLP thành công!")
        
                                                             
        test_text = "hạn chế xe xăng vào trung tâm TP. HCM"
        test_result = rdrsegmenter.word_segment(test_text)
                                                    
        print(f"Test word segment thành công: '{test_text}' -> '{test_result}'")
        print("  VnCoreNLP đã sẵn sàng để xử lý!\n")
    except Exception as e:
        print(f"Lỗi khi khởi tạo VnCoreNLP: {e}")
        import traceback
        traceback.print_exc()
        return
    
                    
    df_articles = process_articles(rdrsegmenter)
    
                    
    df_comments = process_comments(rdrsegmenter)
    
                    
    try:
        rdrsegmenter.close()
    except:
        pass
    
              
    print("\n" + "=" * 60)
    print("THỐNG KÊ")
    print("=" * 60)
    if df_articles is not None:
        print(f"- Số articles đã xử lý: {len(df_articles)}")
    if df_comments is not None:
        print(f"- Số comments đã xử lý: {len(df_comments)}")
    print("\nĐã tạo các file:")
    print("- outputs/preprocessing_step_2_articles.parquet")
    print("- outputs/preprocessing_step_2_comments.parquet")
    print("=" * 60)


if __name__ == '__main__':
    main()
