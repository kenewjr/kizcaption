import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lumacaption.mt.nllb_engine import NllbEngine
from lumacaption.mt.conversational_data import normalize_indonesian_slang, CONVERSATIONAL_MAP, CONVERSATIONAL_ALIASES

# Test cases
TEST_CASES_KBBI = [
    # Salam & Sapaan Formal
    ("Halo", "Hello!"),
    ("Selamat pagi", "Good morning!"),
    ("Selamat malam", "Good evening!"),
    ("Apa kabar?", "How are you?"),
    ("Kabar saya baik.", "My news is good / I am fine."),
    ("Terima kasih banyak.", "Thank you very much."),
    ("Sama-sama.", "You're welcome."),
    ("Maaf mengganggu waktu Anda.", "Sorry to bother you."),
    ("Sampai jumpa besok.", "See you tomorrow."),
    ("Tolong bantu saya menyelesaikan pekerjaan ini.", "Please help me finish this work."),
    
    # Kalimat Deklaratif & Percakapan Standar KBBI
    ("Saya sedang belajar pemrograman komputer di rumah.", "I am studying computer programming at home."),
    ("Hari ini cuaca sangat cerah dan menyenangkan.", "Today the weather is very bright and pleasant."),
    ("Apakah Anda ingin minum kopi bersama saya?", "Do you want to drink coffee with me?"),
    ("Pertemuan akan dimulai tepat pada pukul sembilan pagi.", "The meeting will start at nine in the morning."),
    ("Kami telah menyelesaikan seluruh laporan proyek tepat waktu.", "We have completed the whole project report on time."),
    ("Dia tidak dapat menghadiri acara karena sedang sakit.", "He/she cannot attend the event because he/she is sick."),
    ("Tolong jangan berisik karena bayi sedang tidur.", "Please do not be noisy because the baby is sleeping."),
    ("Buku ini memberikan penjelasan yang sangat mendalam mengenai kecerdasan buatan.", "This book provides a deep explanation about AI."),
    ("Bagaimana cara memperbaiki kesalahan pada sistem ini?", "How to fix errors in this system?"),
    ("Saya sangat menghargai bantuan yang telah Anda berikan.", "I really appreciate the help you have provided."),
    ("Pemerintah daerah sedang membangun jembatan penyeberangan baru.", "Local government is building a new bridge."),
    ("Keluarga kami berencana untuk pergi berlibur ke Yogyakarta akhir pekan ini.", "Our family plans to go on vacation to Yogyakarta."),
    ("Suara musik itu terdengar sangat merdu di malam hari.", "The sound of music sounds very melodious at night."),
    ("Kita harus menjaga kebersihan lingkungan bersama-sama.", "We must maintain environmental cleanliness together."),
    ("Terima kasih atas perhatian dan kerja sama Anda.", "Thank you for your attention and cooperation."),
]

TEST_CASES_GAUL = [
    # Slang Percakapan Harian
    ("Halo guys apa kabar kalian semua?", "Hello guys how are you all?"),
    ("Gue lagi otw ke tempat lu nih.", "I am on the way to your place."),
    ("Lu lagi ngapain sekarang bro?", "What are you doing right now bro?"),
    ("Mager banget gue hari ini sumpah.", "I am so lazy today I swear."),
    ("Jangan baper dong santuy aja kali.", "Don't take it to heart, just relax."),
    ("Kepo banget sih lu jadi orang.", "You are so curious / nosey."),
    ("Nanti malem mabar game bareng yuk.", "Let's play games together tonight."),
    ("Gaskeun bro kita ratain musuhnya.", "Let's go bro let's wipe the enemies."),
    ("Anjir hoki parah lu tadi dapet item langka.", "Wow you were so lucky getting that rare item."),
    ("Kocak banget dah video tadi bikin ngakak.", "That video earlier was so funny it made me laugh."),
    ("Gue gak ngerti apa yang lu omongin barusan.", "I don't understand what you just said."),
    ("Udah makan belom lu? Kalo belom ayo cari makan.", "Have you eaten yet? If not let's find food."),
    ("Makasih banyak ya cuy udah mau nemenin gue.", "Thank you very much bro for accompanying me."),
    ("Gapapa santai aja kawan gak usah buru-buru.", "No problem just relax friend no need to rush."),
    ("Doi kemarin curhat ke gue sampe nangis.", "She/he poured her heart out to me yesterday until crying."),
    ("Gabut parah nih di rumah gak ada kerjaan.", "So bored at home with nothing to do."),
    ("Lu udah dapet tiket konser buat besok belom?", "Have you gotten the concert ticket for tomorrow yet?"),
    ("Yaudah ntar malem kita ketemuan di kafe biasa.", "Alright tonight we will meet at the usual cafe."),
    ("Mantap jiwa konten lu hari ini keren abis.", "Awesome, your content today is super cool."),
    ("Gua mau nyoba fitur baru ini dulu ya guys.", "I want to try this new feature first guys."),
    ("Bikin pusing aja masalah ini gak kelar-kelar.", "This problem is making my head spin, it never ends."),
    ("Keren banget performa live streaming lu barusan.", "Your live streaming performance earlier was so cool."),
    ("Gue belom tidur dari kemarin malem gara-gara ngoding.", "I haven't slept since last night because of coding."),
    ("Sabar bro jangan emosi dulu dengerin penjelasannya.", "Be patient bro don't get emotional listen to the explanation."),
    ("Bye bye semuanya sampai ketemu di live stream berikutnya ya!", "Bye bye everyone see you in the next stream!"),
]

def run_evaluation():
    engine = NllbEngine(
        model_id_or_path="nllb",
        device="auto",
        cache_dir=Path("models/nllb-cache"),
        on_warning=lambda msg: print(f"[LOG] {msg}", flush=True)
    )
    
    print("\n" + "="*70)
    print("EVALUASI TRANSLASI INDONESIA -> ENGLISH (KBBI & GAUL)")
    print("="*70 + "\n")
    
    print("--- 1. UJI BAHASA KBBI / BAKU ---")
    results_kbbi = []
    for text_id, intent_ref in TEST_CASES_KBBI:
        res = engine.translate(text_id, "ind_Latn", ["English"])
        trans_en = res.get("English", "")
        results_kbbi.append((text_id, intent_ref, trans_en))
        print(f"ID   : {text_id}")
        print(f"EN   : {trans_en}")
        print(f"Ref  : {intent_ref}")
        print("-" * 50)
        
    print("\n--- 2. UJI BAHASA GAUL / SLANG / STREAMER ---")
    results_gaul = []
    for text_id, intent_ref in TEST_CASES_GAUL:
        norm_id = normalize_indonesian_slang(text_id)
        res = engine.translate(text_id, "ind_Latn", ["English"])
        trans_en = res.get("English", "")
        results_gaul.append((text_id, norm_id, intent_ref, trans_en))
        print(f"ID (Asli)      : {text_id}")
        if norm_id != text_id:
            print(f"ID (Normalisasi): {norm_id}")
        print(f"EN (Hasil)     : {trans_en}")
        print(f"Ref Intent     : {intent_ref}")
        print("-" * 50)
        
    engine.close()
    return results_kbbi, results_gaul

if __name__ == "__main__":
    run_evaluation()
