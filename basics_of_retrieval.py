# Requirements
# nltk==3.9.1
# numpy==1.26.4
# rank_bm25==0.2.2
# scikit-learn==1.5.2

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

nltk.download("punkt")


# Sample documents about sailing in Croatia
documents = [
    "Sailing in Croatia often includes visiting UNESCO World Heritage sites.",
    "The Makarska Riviera is known for its stunning coastline and sailing opportunities.",
    "Sailing in Croatia offers stunning views of the Adriatic Sea.",
    "The city of Sibenik is home to the impressive St. James's Cathedral, a UNESCO World Heritage site.",
    "The island of Brač is known for its beautiful beaches and great sailing conditions.",
    "Sailing to the island of Rab, known for its medieval old town, is a great experience.",
    "The Pakleni Islands near Hvar are a popular spot for sailing yachts.",
    "The island of Cres is one of the largest in Croatia and a great destination for sailing.",
    "The island of Cres is one of the largest in Croatia and has a diverse wildlife.",
    "The city of Zagreb is the capital of Croatia and offers a mix of modern and historic attractions.",
    "Sailors can experience the traditional Dalmatian way of life in many coastal villages.",
    "The city of Knin is known for its historic fortress and beautiful scenery.",
    "The island of Krk is the largest island in the Adriatic Sea.",
    "Croatia has a rich history dating back to the Roman Empire.",
    "The island of Hvar is a popular destination for celebrities and high-end travelers.",
    "The city of Varazdin is known for its baroque buildings and vibrant cultural scene.",
    "Sailors can enjoy fresh seafood at many coastal restaurants in Croatia.",
    "The coastal town of Senj is known for its carnival and Nehaj Fortress.",
    "The Dalmatian Coast is a famous region for sailing in Croatia.",
    "The Diocletian's Palace in Split is one of the most famous Roman ruins in Croatia.",
    "The island of Dugi Otok is known for its dramatic cliffs and beautiful sailing waters.",
    "The island of Rab is famous for its medieval old town and stunning beaches.",
    "The island of Mljet is home to a national park and is ideal for nature lovers.",
    "Sailors in Croatia can visit ancient Roman ruins in Split.",
    "The city of Kotor, just across the border in Montenegro, is a popular extension for Croatian sailing trips.",
    "The island of Korcula is believed to be the birthplace of Marco Polo.",
    "The town of Motovun in Istria is famous for its film festival.",
    "The city of Nin is known for its ancient salt pans and historic church.",
    "The city of Vukovar is known for its role in the Croatian War of Independence.",
    "Sailing to the island of Lošinj, known for its health tourism, is a relaxing experience.",
    "The island of Hvar is known for its vibrant nightlife and sailing opportunities.",
    "The island of Korčula is believed to be the birthplace of Marco Polo and is a popular sailing destination.",
    "Sailing in Croatia often involves stopping at picturesque fishing villages.",
    "The city of Opatija is known for its grand villas and seaside promenade.",
    "The city of Porec is known for the Euphrasian Basilica, a UNESCO World Heritage site.",
    "The island of Brac is home to the famous Zlatni Rat beach, known for its changing shape.",
    "Sailing around the island of Mljet offers a peaceful and scenic experience.",
    "The Blue Cave on Biševo Island is a must-see for sailors.",
    "The island of Pag is famous for its cheese and nightlife, and is a fun stop for sailors.",
    "The island of Lastovo is one of Croatia's most remote and tranquil destinations.",
    "Sailors can enjoy snorkeling in the clear waters of the Adriatic Sea.",
    "The city of Rovinj is a charming starting point for a sailing trip in Croatia.",
    "Croatia's national parks, like Krka and Plitvice, are ideal for hiking and nature lovers.",
    "The city of Zadar is famous for its sunsets, which sailors can enjoy from the sea.",
    "Croatia has over a thousand islands, each with its unique charm.",
    "Croatia's Adriatic coast is dotted with charming fishing villages.",
    "Sailing to the island of Šolta offers a quiet escape from the more touristy areas.",
    "Sailors in Croatia can explore over 1,200 islands.",
    "Sailing in Croatia provides opportunities to visit ancient fortresses and castles along the coast.",
    "Croatia is famous for its beautiful coastline and crystal-clear waters.",
    "The island of Krk is accessible by bridge and is a popular starting point for sailing trips.",
    "Sailing in Croatia is best enjoyed during the summer months.",
    "The island of Pag is famous for its cheese, which is considered a delicacy.",
    "The medieval town of Rovinj is one of the most picturesque places in Croatia.",
    "The island of Losinj is famous for its health tourism and clean air.",
    "The city of Dubrovnik, with its famous city walls, is a top destination for sailors.",
    "Croatia has beautiful islands perfect for sailing.",
    "Sailing in Croatia allows you to explore hidden coves and bays.",
    "The city of Rijeka is an important cultural and economic center in Croatia.",
    "The island of Mljet has a national park that is perfect for exploring by sailboat.",
    "The Istrian Peninsula is famous for its truffles and gourmet food.",
    "The town of Cavtat is a quieter alternative to nearby Dubrovnik.",
    "The Peljesac Peninsula is known for its vineyards and wine production.",
    "The Dubrovnik Summer Festival is a major cultural event featuring theater, music, and dance performances.",
    "The city of Zadar is famous for its unique Sea Organ, an architectural sound art object.",
    "The beaches in Croatia are among the best in Europe, with many receiving Blue Flag status.",
    "Sailing around the island of Murter gives access to the Kornati Islands National Park.",
    "The city of Trogir, with its historic architecture, is a great place to dock.",
    "Sailing from Split to Dubrovnik offers breathtaking coastal scenery.",
    "The Plitvice Lakes National Park is a UNESCO World Heritage site known for its stunning waterfalls and lakes.",
    "The city of Hvar is one of the sunniest places in Europe and a popular sailing hub.",
    "The city of Dubrovnik is often called the 'Pearl of the Adriatic'.",
    "The city of Osijek is located in the eastern part of Croatia and is known for its Baroque style.",
    "The city of Pula, with its Roman amphitheater, is a unique sailing destination.",
    "The town of Trogir is a UNESCO World Heritage site known for its medieval architecture.",
    "Sailors in Croatia can enjoy local wines at many coastal vineyards.",
    "Croatia has a Mediterranean climate, making it a great destination year-round.",
    "Croatia has a diverse cultural heritage, with influences from Italy, Hungary, and Austria.",
    "The city of Rijeka is an important cultural and historical sailing destination.",
    "The best sailing routes in Croatia include Dubrovnik and Split.",
    "The Brijuni Islands are a national park and a former presidential retreat.",
    "Sailing to the Elaphiti Islands offers a mix of natural beauty and cultural sites.",
    "Croatia is known for its delicious seafood cuisine.",
    "The ancient city of Pula is known for its well-preserved Roman amphitheater.",
    "The city of Opatija is a historical seaside resort that welcomes sailors.",
    "The waters around Croatia are known for being calm and clear, ideal for sailing.",
    "The Kornati Islands National Park is a popular sailing destination in Croatia.",
    "Sailing around the Brijuni Islands offers a glimpse of Croatia's natural beauty and wildlife.",
    "The island of Vis was a military base and was closed to tourism until the 1990s.",
    "The city of Split is a major port and gateway to the Dalmatian islands.",
    "Sailing to the island of Vis provides access to the famous Blue Cave.",
    "The city of Buzet in Istria is known as the 'City of Truffles'.",
    "The city of Karlovac is known for its parks and the rivers that flow through it.",
    "Sailing to the island of Vis allows you to experience a more remote part of Croatia.",
    "The city of Šibenik is a UNESCO World Heritage site and a great stop for sailors.",
    "The town of Samobor is famous for its traditional cream cake called 'kremsnita'.",
    "Sailing around the island of Lastovo provides a more secluded experience.",
    "Croatia's wine regions produce some excellent wines, especially in Istria and Dalmatia.",
    "Many sailors start their Croatian adventure from the city of Zadar.",
    "The coastal town of Makarska is known for its beautiful beaches and lively nightlife.",
]

# Tokenize words
sample_document = documents[0]
tokenized_words = nltk.word_tokenize(sample_document)

print(tokenized_words)


# Tokenize sentences
sample_document = " ".join(documents[0:2])
tokenized_sentences = nltk.sent_tokenize(sample_document)

print(tokenized_sentences)
