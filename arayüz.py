import joblib
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

def predict_price(location, room, area, age, floor):
    try:
        user_data = pd.DataFrame({
            'Room': [room],
            'Area': [area],
            'Age': [age],
            'Floor': [floor],
        })

        location_encoded = encoder.transform(pd.DataFrame([location], columns=['Location']))
        location_df = pd.DataFrame(location_encoded, columns=encoder.get_feature_names_out(['Location']))

        user_data = pd.concat([user_data, location_df], axis=1)

        # Tahmin yap
        price_prediction = model.predict(user_data)
        return price_prediction[0]
    except Exception as e:
        messagebox.showerror("Hata", f"Tahmin sırasında bir hata oluştu: {str(e)}")
        return None

# Model ve Encoder'ı yükle
model = joblib.load('best_model.pkl')
encoder = joblib.load('encoder.pkl')


df = pd.read_csv('/mnt/c/Users/burak/OneDrive/Masaüstü/EV/istanbul_satilik_evler_2023.csv')
df['Room'] = df['Room'].apply(lambda x: eval(str(x)) if isinstance(x, str) else x)
min_area = df['Area'].min()
max_area = df['Area'].max()
min_floor = df['Floor'].min()
max_floor = df['Floor'].max()
min_age = df['Age'].min()
max_age = df['Age'].max()
min_room = df['Room'].min()
max_room = df['Room'].max()
# Tkinter penceresi oluştur
root = tk.Tk()
root.title("Ev Fiyat Tahmini")
root.geometry("400x400")

# Mevcut konumları al
locations = encoder.categories_[0]


# Etiketler ve giriş alanları
tk.Label(root, text="Semt").grid(row=0, column=0, pady=5)
location_combobox = ttk.Combobox(root, values=locations, state="readonly")
location_combobox.grid(row=0, column=1, pady=5)

tk.Label(root, text="Oda Sayisi").grid(row=1, column=0, pady=5)
room_entry = tk.Entry(root)
room_entry.grid(row=1, column=1, pady=5)

tk.Label(root, text="Ev Genisligi (m2)").grid(row=2, column=0, pady=5)
area_entry = tk.Entry(root)
area_entry.grid(row=2, column=1, pady=5)

tk.Label(root, text="Ev Yasi").grid(row=3, column=0, pady=5)
age_entry = tk.Entry(root)
age_entry.grid(row=3, column=1, pady=5)

tk.Label(root, text="Kat").grid(row=4, column=0, pady=5)
floor_entry = tk.Entry(root)
floor_entry.grid(row=4, column=1, pady=5)

# Tahmin fonksiyonu
def calculate_price():
    location = location_combobox.get()
    if not location:
        messagebox.showwarning("Uyari", "Lutfen bir semt secin!")
        return

    try:
        room = int(room_entry.get())
        area = float(area_entry.get())
        age = int(age_entry.get())
        floor = int(floor_entry.get())
    except ValueError:
        messagebox.showerror("Hata", "Lutfen tüm verileri dogru formatta girin!")
        return

    # Sınır kontrolü
    if room < min_room or room > max_room:
        messagebox.showwarning("Uyari", f"Oda sayisi {min_room} ile {max_room} arasinda olmalidir!")
        return
    if area < min_area or area > max_area:
        messagebox.showwarning("Uyari", f"Ev genisligi {min_area} m² ile {max_area} m² arasinda olmalidir!")
        return
    if age < min_age or age > max_age:
        messagebox.showwarning("Uyari", f"Ev yasi {min_age} yil ile {max_age} yil arasinda olmalidir!")
        return
    if floor < min_floor or floor > max_floor:
        messagebox.showwarning("Uyari", f"Kat sayisi {min_floor} ile {max_floor} arasinda olmalidir!")
        return

    
    price = predict_price(location, room, area, age, floor)
    if price is not None:
        formatted_price = "{:,.2f}".format(price).replace(",", ".").replace(".", ",", 1) + " TL"
        messagebox.showinfo("Tahmin Edilen Fiyat", f"Bu ozelliklere sahip evin tahmin edilen fiyati: {formatted_price}")


tk.Button(root, text="Fiyat Tahmin Et", command=calculate_price).grid(row=5, column=0, columnspan=2, pady=10)

root.mainloop()