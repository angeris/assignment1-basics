# %%
from cs336_basics.tokenizer import Tokenizer
import pickle
import time
import sys

# %%
stories = """
u don't have to be scared of the loud dog, I'll protect you". The mole felt so safe with the little girl. She was very kind and the mole soon came to trust her. He leaned against her and she kept him safe. The mole had found his best friend.
<|endoftext|>
Once upon a time, in a warm and sunny place, there was a big pit. A little boy named Tom liked to play near the pit. One day, Tom lost his red ball. He was very sad.
Tom asked his friend, Sam, to help him search for the ball. They looked high and low, but they could not find the ball. Tom said, "I think my ball fell into the pit."
Sam and Tom went close to the pit. They were scared, but they wanted to find the red ball. They looked into the pit, but it was too dark to see. Tom said, "We must go in and search for my ball."
They went into the pit to search. It was dark and scary. They could not find the ball. They tried to get out, but the pit was too deep. Tom and Sam were stuck in the pit. They called for help, but no one could hear them. They were sad and scared, and they never got out of the pit.
<|endoftext|>


Tom and Lily were playing with their toys in the living room. They liked to build towers and bridges with their blocks and cars. Tom was very proud of his tall tower. He wanted to make it even taller, so he reached for more blocks.
"Tom, can I have some blocks too?" Lily asked. She wanted to make a bridge for her cars.
"No, these are mine. Go find your own," Tom said. He did not want to share with his sister. He pulled the blocks closer to him.
Lily felt sad and angry. She did not think Tom was being nice. She looked at his tower and had an idea. She decided to pull one of the blocks at the bottom of the tower.
Suddenly, the tower fell down with a loud crash. All the blocks and cars scattered on the floor. Tom and Lily were shocked. They felt the floor shake and heard a rumble. It was an earthquake!
"Mommy! Daddy!" they cried. They were scared and ran to their parents, who were in the kitchen.
"Are you okay, kids?" Mommy asked. She hugged them and checked if they were hurt.
"We're okay, Mommy. But our toys are broken," Lily said.
"I'm sorry, Lily. But toys are not important. You are important. We are safe and together. That's what matters," Mommy said.
Tom felt sorry for what he did. He realized he was selfish and mean to his sister. He saw how scared she was during the earthquake. He wanted to make her happy.
"Lily, I'm sorry I did not share with you. You can have all the blocks you want. I love you, sister," Tom said.
Lily smiled and hugged him. She forgave him and thanked him. She loved him too.
They went back to the living room and cleaned up their toys. They decided to build something together. They made a big house with a garden and a fence. They put their cars and dolls inside. They were happy and proud of their work.
Mommy and Daddy came to see their house. They praised them and gave them a treat. It was a lemon cake. It was sour, but they liked it. They learned that sharing is caring, and that family is sweet.
<|endoftext|>

Once upon a time there was a little girl named Lucy. She loved to go to the store to buy sweets with her mom and dad. On this special day, Lucy entered the store with her mom and dad, feeling so excited.
As they were looking around, Lucy noticed a little girl playing with a toy in the corner of the store. She gasped in excitement and ran towards her. Lucy asked if she could play too but the little girl said no. She was rather grumpy and was not in the mood to play.
Lucy's mom saw what was going on and told Lucy, "Let's try to be peaceful and kind to her. Have patience and understanding. Together, you can both be happy!"
So, Lucy smiled at the girl and said, "Can we play together?" The little girl softened and smiled back. She agreed to share the toy and even let Lucy have a turn first.
Lucy and the little girl played together happily. In the end, they both learnt an important lesson: be peaceful, kind, and understanding when faced with a conflict. And that is why Lucy and the little girl became great friends.
<|endoftext|>
One morning, a cat named Tom woke up. He felt happy because the sun was shining. Tom wanted to start his day, so he did a big stretch. He stretched his legs, his back, and his tail. It felt easy and good.
Tom went outside to play. He saw his friend, a dog named Max. Max was also stretching in the morning sun. They both felt very happy. They decided to play together and have fun all day.
At the end of the day, Tom and Max were tired. They had played all day and had lots of fun. They said goodbye to each other and went to their homes. Before going to sleep, they both did another easy stretch. Tom knew that tomorrow would be another happy morning.
<|endoftext|>


Lily and Tom were twins who liked to decorate things. They had a big box of crayons, stickers, and glitter. One day, they found a shiny copper pot in the kitchen. It was Mom's pot, but she was not home. Lily and Tom wanted to make it more pretty.
They took the pot to their room and put it on the floor. They opened their box of crayons, stickers, and glitter. They started to draw and stick and sprinkle on the pot. They made colorful shapes and patterns. They thought the pot looked very nice.
But they were clumsy. They did not see that they also made a big mess. They spilled glitter on the floor and the bed. They stuck stickers on the wall and the door. They drew crayons on the window and the dresser. They did not hear Mom come home.
Mom saw the mess in the kitchen. She saw the glitter, the stickers, and the crayons. She was angry. She followed the trail to their room. She saw the pot. She saw the floor, the bed, the wall, the door, the window, and the dresser. She was very angry.
She said, "Lily and Tom, what did you do? You ruined my pot and my room. You are very naughty. You have to clean up everything. And you have to say sorry."
Lily and Tom were scared. They did not mean to make Mom angry. They only wanted to decorate the pot. They said, "Sorry, Mom. We love you. We will clean up. Please don't be mad."
Mom sighed. She was still angry, but she also loved them. She said, "I love you too, but you have to be careful. You can't touch my things without asking. And you can't make a mess like this. You have to learn to be more tidy and respectful."
Lily and Tom nodded. They hugged Mom and said, "We will, Mom. We will." They took a broom, a dustpan, and a cloth. They started to clean up their mess. They hoped Mom would forgive them. They learned their lesson. They would not decorate Mom's pot again.
<|endoftext|>

Once upon a time, there was a king. He was a big and strong king who ruled over his kingdom. One day, he wanted to take a nice and long bath, so he filled up his big bathtub with warm water. He wanted to feel relaxed and so he soaked in the tub for a really long time.
When he had finished soaking and stepped out of the bathtub, the king noticed that the water had spilled out of the tub and all over the floor. He felt guilty that he had made such a mess, so he quickly grabbed a cloth and began to clean it up.
The king got so hot from cleaning up the mess that he decided to take another soak in the bathtub. He put a lot of bubbles in the water to make it nice and bubbly. He relaxed again and felt all the worries wash away.
The king was so happy that he had been able to clean up the mess he had made and enjoy a nice soak. He dried off and wrapped himself up in a big towel. Then, the king went back to ruling his kingdom and enjoying his lovely baths.
<|endoftext|>


Lily and Max were playing in the park with their mom. They liked to slide, swing, and run on the grass. They also liked to listen to the birds that made whistles in the trees.
"Look, mom, a red bird!" Lily said, pointing to a cardinal.
"That's a pretty bird, Lily. Do you know what it is called?" mom asked.
"A cardinal, mom. I learned it in school," Lily said proudly.
"Very good, Lily. And do you know what that yellow bird is, Max?" mom asked, pointing to a canary.
"A canary, mom. I learned it in school, too," Max said.
"Wow, you are both very smart. Do you want to learn another bird name?" mom asked.
"Yes, mom, yes!" Lily and Max said.
"OK, see that blue bird over there? That's a blue jay. It has a very loud whistle. Can you try to whistle like it?" mom asked.
Lily and Max tried to whistle, but they only made funny noises. They laughed and mom laughed, too.
"Whistling is hard, mom. How do you do it?" Lily asked.
"It takes practice, Lily. Maybe when you are older, you can whistle better. But you know what? You don't need to whistle to have fun. You can sing, or clap, or dance, or make any sound you like," mom said.
"I like to sing, mom. Can we sing a song?" Max asked.
"Sure, Max. What song do you want to sing?" mom asked.
"How about 'Twinkle, Twinkle, Little Star'?" Max suggested.
"OK, let's sing it together," mom said.
They sang the song and looked at the sky. The sun was shining and the clouds were light and fluffy.
"That was a nice song, mom. But I'm feeling sleepy now. Can we nap?" Lily asked.
"Me too, mom. Can we nap?" Max asked.
"Of course, my sweeties. Let's go to the car and nap. You had a busy day," mom said.
They walked to the car and mom buckled them in their seats. She gave them each a kiss and a hug.
"Sleep well, my loves. I'll wake you up when we get home," mom said.
Lily and Max closed their eyes and fell asleep. They dreamed of birds and stars and whistles. They were happy.
<|endoftext|>
Once upon a time, there was a big bow. The bow was very strong and reliable. It was the best bow in the town. Everyone liked the bow and wanted to use it. They knew it would help them do their work.
One day, a man wanted to test the bow. He was not a good man. He wanted to see if the bow was really strong. He pulled and pulled on the bow. He wanted to see if it would break.
The bow did not break because it was strong. But the man did not stop. He pulled harder and harder. At last, the bow broke. The man was not happy. The town was sad. They lost their best bow.
<|endoftext|>
Once upon a time, there was a little girl named Lily. She lived in a small, tidy house with her mom, dad, and her dog, Max. Lily loved to play with Max in the backyard. They would run, jump, and have lots of fun together.
One day, Lily's mom said, "Lily, I have a special treat for you and Max!" She gave Lily a big, yummy cookie and Max a tasty bone. Lily was very happy and said, "Thank you, mom!" But then, she had an idea. She wanted to save the cookie and the bone for later.
Lily put the cookie and the bone in a secret place under her bed. She forgot about them for a few days. When she remembered the treats, she found that the cookie was all broken and the bone was dirty. The treats were spoiled. Lily was sad, but she learned that it's better to enjoy treats when they are fresh and clean.
<|endoftext|>
Once upon a time, in a small house, there lived a little boy named Tim. Tim was an intelligent boy who loved to learn new things. One day, he found a big book about ghosts. He read the book and learned that ghosts could be nice or scary.
One day, Tim met a friendly ghost named Gigi. Gigi was lost and needed help. Tim wanted to show Gigi that he was a good friend. So, he picked up Gigi's favorite toy, a round ball. But, as he was playing with the ball, he accidentally let it drop. The ball broke into pieces. Tim felt sad and said sorry to Gigi.
Gigi smiled and told Tim that it was okay. She said that everyone makes mistakes. The important thing is to say sorry and learn from them. Tim learned that being a good friend means helping each other and being kind. And from that day on, Tim and Gigi became the best of friends.
<|endoftext|>
Once upon a time, in a big forest, there was a wise old owl. This owl had a long neck and knew many things. He lived in a tall tree and helped all the animals in the forest.
One day, a little bird came to the wise owl. The bird was sad because his nest was broken. The wise owl wanted to help restore the nest. They worked together to fix the nest, using sticks and leaves.
But as they were fixing the nest, a big wind came. The wind blew the nest away, and the wise owl's long neck got stuck in the tree. The little bird was sad, and the wise owl was stuck. They could not restore the nest, and the wise owl's neck hurt. The forest was not happy that day.
<|endoftext|>
Once upon a time, there was a little girl named Sue. Sue was very thoughtful. She always helped her mom and dad. One day, Sue saw her mom trying to open a door with a broken handle. Sue wanted to help her mom.
Sue asked her mom, "Can I help you?" Her mom said, "Yes, Sue. We need a new handle for the door. Can you ask dad if he has one?" Sue went to her dad and asked, "Dad, do we have a new handle for the door?" Her dad looked at Sue and said, "I am not sure, let's look together."
Sue and her dad looked for a new handle. They found one, but it was very high up. Sue's dad tried to reach it, but he couldn't. Sue had an idea. She said, "Dad, let's use a chair to stand on." Her dad refused. He said, "No, Sue. That is not safe. Let's ask mom for help." So, they asked mom for help, and she found a safe way to get the handle. They fixed the door together, and Sue felt happy that she could help her mom and dad.
<|endoftext|>
"""
with open("output/bpe_tinystories.pkl", "rb") as f:
    tinystories_bpe = pickle.load(f)

tinystories_special_tokens = ["<|endoftext|>"]

tinystories_tokenizer = Tokenizer(tinystories_bpe['vocab'], tinystories_bpe['merges'], tinystories_special_tokens)

tiny_stories_encoding = tinystories_tokenizer.encode(stories)

bits_per_token = 16
total_size = bits_per_token*len(tiny_stories_encoding)

print(f"Total size: {total_size} with {bits_per_token} bits per token")
original_size = len(stories.encode())*8
print(f"Original size {original_size}")

owt_text = """
LOUISVILLE, Ky. — A few unflattering reviews are to be expected with any hotel, particularly one whose rates start at $49 per night. But while complaints about shabby rooms and thin towels are common in the industry, ones like these, from TripAdvisor.com, are not: “It is a clean hotel but there are a lot of homeless people there.” “Run far far away!!!!! This is a homeless shelter, not a hotel!” “DO NOT STAY HERE UNLESS YOU ARE HOMELESS… All of the workers are former addicts/homeless people.” Hotel Louisville, 12 stories of brick adorned with a large white cross, is indeed a hotel and event space open to the public. At the same time, it is a transitional-housing facility, substance-abuse recovery center and job-training site owned and operated by Wayside Christian Mission, a nonprofit that shelters and feeds the city’s homeless population. Wayside bought the building at a foreclosure auction in 2009, never intending to rent rooms to the general public. It was simply a place to house the homeless. But as expenses mounted and travelers came through the lobby, remembering what used to be a Holiday Inn and seeking a place to stay, Wayside began to make use of its empty rooms. Four years later, Hotel Louisville is in many ways an improbable success, serving addicts and the homeless while turning a profit from hotel guests and banquets, even during the recession. Perhaps the nation’s only such hybrid, it defies the usual categories — homeless shelter and charity; hotel and for-profit enterprise — and reflects a growing embrace of commerce by social-services groups normally funded by government and foundation grants. Yet Wayside’s pivot from a traditional model of charity toward the seductions of business tells its own complicated tale, showing just how hard it is to do good.

Not In My Backyard

The lobby of Hotel Louisville Pat McDonogh for Al Jazeera America Every homeless shelter has a NIMBY problem. Try building a new facility or renovating an old one and the neighbors come out of the woodwork to protest each additional bed. But the battle waged against Hotel Louisville was unusual even in the long history of Wayside Christian Mission, founded in 1957. The saga began six years ago, after the group finally raised enough money to replace its worn-out transitional-housing facility for women and kids. Initially, the married couple at Wayside’s helm — Tim Moseley, a bearded, heavyset minister, and his wife, Nina, an attorney with waist-length platinum blonde hair — intended to build on property it already owned along gentrifying Market Street. Real-estate developers with city-hall ties killed the plan, claiming the need for “historic preservation,” and forced Wayside to sell its Market Street building. The Moseleys then tried to buy a former school, but that effort was blocked by irate neighbors and a zoning decision effectively prohibiting new homeless shelters in the city. The ban was later declared unlawful by the federal Department of Housing and Urban Development.

Then, in early 2009, the Moseleys heard that the downtown Holiday Inn, nicknamed “Hotel Louisville,” would be sold at a foreclosure auction. The final price tag of $10 million depleted all the funds Wayside had raised through its years-long capital campaign and proceeds from the Market Street sale, but at 187 rooms and 169,400 square feet, the building could house hundreds. Eighty-three homeless women moved into the hotel in November. Shortly thereafter, with utility costs mounting and many floors vacant, the Moseleys saw an opportunity. “People kept coming through and asking for a room,” Nina Moseley recalled. So Wayside opened Hotel Louisville to the public while continuing to provide shelter and substance-abuse recovery services to women in need, free of charge.

Home at a hotel

The hotel’s high-ceilinged lobby resembles that of a ski resort, with an elaborate chandelier, a grand piano and extensive wood paneling. It’s a busy crossroads of guests and long-term residents — plus thrifty diners lured by the $5 soul food-buffet in the adjacent restaurant. I first visited Hotel Louisville in June and recently returned for a five-day stay. After checking in at the front desk, I took a well-worn elevator to my room on the ninth floor, one of five levels open to the general public. It was a standard, budget-rate hotel room: clean and air-conditioned, with a TV and small desk free of flourish.

Cassie Lintz settles down for the night with her daughters Kendal, age 4, and Chloe, age 6, on right. Pat McDonogh for Al Jazeera America I’d soon learn that the residents’ quarters, on floors four, five, six and eleven, are homey and personalized, more like tiny apartments than hotel rooms sanitized with Smells BeGone. To date, between 96 and 162 people at a time — mostly women in recovery, but also some children and a few men — have lived in the hotel. Cassie Lintz and her daughters, preschooler Kendal and first-grader Chloe, know the place better than most. Several months ago, they moved in for a second time. “I had almost two years clean when I relapsed. So I came back here to do this again and get back on track,” Lintz said of her struggle with prescription painkillers.

She and the two round-faced, sandy-haired girls were sharing a room on the hotel’s family floor, home to several mostly single-parent units. Generally, substance-abuse centers require parents to come alone, leaving their children behind. Hotel Louisville is a rarity even among family-based recovery programs, giving clients with children a free, private room, with child care and activities included. Upstairs, on one of the singles floors, Yolanda Thomas wore her reading glasses to study the Bible in bed. She’d arrived at Hotel Louisville in July, having slept off her last high on a bus from Virginia. She was living in a tidy double with another “girl in the program,” splitting a bathroom, nightstand, TV and table. Thomas’s shoes — including several pairs of pointy high heels she has little occasion to wear — were lined up along the wall.

Recovering through work?

At early-morning meditation on a recent Friday, Thomas and about two dozen other women — in varying degrees of wakefulness, young and old, white and African-American — sat in a circle of chairs in the first-floor chapel. Everyone clutched a Bible and The Big Book from Alcoholics Anonymous. A few brought young children, who played in the back or snoozed in their strollers. (The day care wasn’t yet open.) Hotel manager Virginia Taylor, or “Miss V,” entered the room around 8 a.m. Short and stocky with freckled brown skin and oiled hair, she’s a former addict with 22 years clean and nearly as much peer-counseling experience. Taylor oversees Wayside’s recovery program and leads the gospel choir. In a thick Southern accent full of tapering R’s, she speaks with the air of a preacher or prison warden. “The disease does not promise you it’ll be easy. But you have a daily choice to stay sober. Choose to live,” she said. Taylor led the community meeting, a chance for residents to express grievances with the facility and with one another. Women clapped for every submission of “consequences” — 10,000-word reflections written on handfuls of notebook paper — the punishment for minor infractions, like missing a work shift or neglecting to sign out. The meeting concluded with a collective Lord’s Prayer, and the women then split up for work. Work therapy — performing housekeeping, food service, security and laundry for public hotel guests — is a key component of Wayside’s recovery program.

Recovery Manager Virginia Taylor, a former addict, leads early-morning meditation, part of the substance-abuse curriculum at Hotel Louisville. Pat McDonogh for Al Jazeera America “Many of these ladies didn’t know anything about housekeeping, but here, once you get to second phase (of recovery), they can choose to go to another hotel and get a job,” Cheri Hartwill, a recovering addict who leads the cleaning team, explained. “It prepares you — your work ethic, how you treat people — because a lot of us come from the streets.” Since the work is part of their recovery, and because lodging, food and other basic needs are provided free of charge, hotel residents are given an hourly “gift” of 50 cents to $1.50 per hour in lieu of pay. Residents sign “contracts acknowledging that they’re not employees or workers, but trainees,” said Wayside Chief Operating Officer Nina Moseley. The work assignments at Hotel Louisville are a more elaborate version of what Wayside and other charities have always asked of their residents. Since long before the hotel opened, Wayside has operated two donation-based Louisville thrift stores, with revenues supporting the organization’s central work. It based this model on that of larger groups like Goodwill and the Salvation Army, which require “beneficiaries” to unload, sort, stock and inventory donated goods in vast warehouses and shops. But Hotel Louisville residents are split on the value of work therapy. “It’s part of my penance,” Kim, a former nurse, said of her time in the hotel laundry room. Before coming to Wayside in September, she’d lost her home and car and had to sleep on the porch of an abandoned house. Others in the program say that their long hours and the strenuous nature of work therapy sometimes make it difficult to attend meetings and classes. One resident, who asked to remain anonymous, called these work assignments "unpaid labor," plain and simple. “This isn’t training. This is a business,” she said. “These are actual jobs you’re performing. You’re not doing it to strengthen yourself for a job outside. This is a job inside.” Karen Garnett, district director of the federal Department of Labor in Louisville, questions whether Hotel Louisville’s work-therapy program complies with U.S. labor laws. In some cases, such “trainees” are in fact employees entitled to the minimum wage and overtime, she said. “For the part of the hotel that’s a business entity, there may be some part that could be considered training, but it takes a minimum amount of training for some of this work.”

Manager Linda Stith went from being a homeless addict to helping run Hotel Louisville. Pat McDonogh for Al Jazeera America Wayside stands by its “training hotel” model, which its directors say teaches critical occupational and life skills to clients long removed from the labor force. “We have a very good rate of success for people who complete their programs,” Nina Moseley said. While Wayside does not track information on recovery participants (data collection has not been a priority), many graduates have gone on to work in local hotels, restaurants and hospitals. The hotel cannot afford to pay minimum wage to work-therapy participants, said Linda Stith, one of three general managers. The commercial side of the hotel has subsidized Wayside’s charitable programs since the summer of 2011. Last year, net proceeds of approximately $258,000 — from hotel guests, banquets and restaurant income — funded not only the residential recovery program but also Wayside’s other facility: a traditional homeless shelter and soup kitchen on nearby Jefferson Street. Altogether, the organization’s annual budget for homeless services and recovery is about $3.4 million, much of which comes from government contracts and private donations.

Business or charity or both

Still, not everyone knows the hotel’s backstory, and as its quality improved, Hotel Louisville began to draw patrons unfamiliar with Wayside. “You’d have guests come in. They’d see our clients and say, ‘This is a shelter!’ so I was constantly explaining, ‘This is more than (that). It’s a training hotel,’” general manager Stith said. Wayside’s success, born of experimentation, has garnered the support of philanthropists and other nonprofits, as well as hospitality experts, local politicians, business leaders and scholars. “Tim and Nina Moseley are walking saints,” said Keith Lermie, dean of the hospitality school at Louisville’s Sullivan University. “I’ve seen them do a very good job, and I’ve been in the hotel-restaurant business my whole life. There’s no one downtown providing a safe, clean room at ($49 a night, and) there are a lot of groups that couldn’t afford to have a banquet anywhere else.” And in a climate of dwindling resources for the poor, Hotel Louisville is part of an entrepreneurial trend among social-services nonprofits. For example, Seattle-based FareStart, a recovery and job-training program for homeless and at-risk adults, generates 50 percent of its operating revenue from food-service enterprises staffed by client-trainees. In New York City, the supportive-housing provider Common Ground, which Nina Moseley cites as an early inspiration, covers a growing percentage of its budget with facility rentals and a tax-credit advisory business for nonprofit housing developers. “The fact is, there are fewer dollars to go around,” said Brenda Rosen, Common Ground’s executive director. “There has to be other ways (besides government and foundation grants) to provide services, because the need just continues to grow.” The trick for groups like Common Ground, FareStart and Hotel Louisville, Rosen said, is to avoid putting business first.

I was still under the impression that I was different. I wasn’t willing to identify with homeless women. Manager and former client of Hotel Louisville

A new, professional identity

On the evening of Oct. 12, couples and families dressed to the nines — in gowns, platform heels, hats, suits and tuxedos — filed into Hotel Louisville for a fashion show and banquet on the second floor. It was the annual fundraising gala for a local African-American scholarship fund. “We always have our gala here,” a well-coiffed woman said. She had no idea that the hotel was anything other than a hotel. Three "recovery girls” — Pam Allen, Millie Morris and Ramonica Kellam — were on serving duty. Wearing tuxedo vests, they carried trays of drinks through the crowded ballroom, around chattering guests and the perimeter of the catwalk, to and from the kitchen.

Restaurant manager Kevin Nelson serves dinner at an evening banquet. Pat McDonogh for Al Jazeera America Dinner was served buffet style, and General Manager Kevin Nelson was among those dishing out hot, fragrant Southern food. Nelson, a single father with a bushy mustache and warm, polite manner, lives with his son and daughter in adjoining rooms of Hotel Louisville. He had lost his job as a restaurant manager and fallen on hard times when he came to Wayside in 2011. “In my mind I was just going to work and get myself situated and then get another job somewhere else,” he said. “They were just trying to get the banquets going, and the kitchen and restaurant, and I had done this for close to 22 years, so in a weird way, it all worked out.”<|endoftext|>Years had led up to this. Countless hours of training with her team, private sparring with Blake, all those boring classes, everything. Roman Torchwick was going down, tonight. Yang was running through the twisting hallways underneath the factory in which he was holed up. Ruby had run ahead while she and Blake had dealt with Adam. Yang knew that they had a past together and had left her to make peace. Now all she was focusing on was catching up with her sister.

Yang cursed under her breath, she never should have let her go off on her own. She shook her head lightly. No Ruby was fully capable, she shouldn't be worried. Yang rounded a corner and dug her heels in sliding to a stop. Ruby was standing in the middle of the room, face to face with Roman.

"She got him!" Yang thought her heart filling with pride.

All of that pride turned to dread. Ruby's head was tilted back. She wasn't holding her own weight. Suddenly, Roman dropped her. Her body hit the concrete floor hard, the knife in her chest now in plain sight. Yang stood there, speechless, looking at the limp body of her baby sister.

Roman laughed, "Too late." He said coldly, with a laugh.

Yang saw red, her eyes closed and she could feel the air around her beginning to catch fire. Rational thought dissolved into madness and she screamed, a blood curdling howl of sheer agony. She fired her gauntlets and sailed across the room, her fist catching Roman in the right shoulder. She felt bone break. Her punch flung him across the room and into the far wall. Yang fired again and closed the distance between them and started punching. Tears and sweat mixed with fire and evaporated. She felt Roman struggle, she kept punching. She felt him stop struggling, she kept punching. She felt her fists hitting concrete. She kept punching.

"Yang!" a voice shrieked. She was pulled back into sanity. She looked and saw Blake in the doorway, her hands covering her mouth and fear in her eyes. Her eyes darted back to Roman and she gasped. Where he used to be there was a section of broken wall, blood, and burnt flesh. He was dead, she had killed him. Yang fell back into her hands, crawling away from what was left of him in horror of what she had done.

A few feet away she stopped and just stared blankly at the aftermath, her breathing erratic and shallow. Something grabbed her wrist. She jumped. And looked at back. Ruby looked up at her, there were tears in her eyes.

"Did we do it? Did we get him?"

Yang flipped around on her hands and knees crouching over her little sister. She choked back a sob, "yeah, we got him sis." Her tears fell onto Ruby's shoulder. Ruby looked at her sadly.

"Don't cry. We…." Ruby's eyes drifted as she trailed off.

"NO!" Yang cried.

"Yang, I'm scared." She could barely whisper. Ruby coughed and blood came up.

Ruby looked back to her "I love you… Yang, I love…" Her eyes lost focus and her fingers went limp around Yang's wrist.

Yang's eyes filled with tears again and she fell, burying her face in Ruby's chest. She grasped for her, grapping her cloak and balling it up in her fists. She let the tears flow freely. She couldn't be gone. She couldn't be dead. Yang pulled back throwing her head back and screaming. She raged at the world, for taking her, and she raged at herself for allowing it. Her scream broke off into hollow sobs. She looked down at Ruby taking her hands in her own. She folded her hands over her, covering the knife. She looked up at Blake who was still in the same position. The fear was gone from her eyes and replaced with pain

She slid her arms underneath Ruby and lifted her up, holding her as close as possible. Yang began to walk back, her face devoid of emotion, though her eyes betrayed her. She could feel Blake shadowing her the entire time, silent. Yang wished she would say something, she wished that Blake knew how much she needed her. The walk out of the warehouse didn't feel real. This couldn't be real. This was every single one of her nightmares come true at once. Ruby dead, Blake distant. Yang's eyes widened as it dawned on her. Ruby's body melted away, and she felt Blake's arms slide around her waist. Yang woke up.

"Morning." Blake whispered in her ear, kissing her cheek gently.

Yang spun around grabbing Blake as tightly as possible, burying her face into her shoulder. Blake jumped a little, startled.

"Are you ok, Yang?" she asked concerned.

"Nightmares." Yang said curtly, muffled by her hair.

Blake's tone softened "The same one? With Ruby?" she asked, bringing her arms fully around Yang's back rubbing between her shoulder blades reassuringly.

Yang started to tear up again. "I know I shouldn't be worried, I know she has her own team now. But what if she does get hurt. What if she does…" Yang trailed off, not being able to finish the thought.

"She won't." Blake said firmly, "But no matter what, I will be here for you. I promise."

Yang looked up at Blake and kissed her gently. Blake kissed her back, but quickly pulled back.

"We can't stay in bed all day, we have stuff to do." Blake reminded her.

Yang groaned "Uhg. I know." She rolled out of bed and stumbled tiredly over to the dresser, and began putting on clothes for the day. Halfway through sliding her shirt on, Blake hugged her from behind.

"Why don't you take the day off? You seem like you could use some peace and quiet." Blake suggested.

"Are you sure, we have practice today in Forever Fall?" Yang asked.

"I'll do some tracking with them today. You don't need to be there for that." Blake assure her.

Yang sighed and slid her shirt back off, turning around to face Blake. "You're the best." She said, hugging her. She crawled back into bed and turned to watching Blake get ready. After finishing, Blake picked Gambol's Shroud off of the Nightstand, winked at Yang, turned, and walked out the door without a word.

So much had changed since last year. The war between Vale and another kingdom had forced their training to be accelerated drastically. Teams had been broken in half, and new recruits used to fill the empty spots. Yang was very happy to have been paired with Blake. While she missed her sister greatly, without the current arrangement her and Blake never would have become this close. Constant nightmares had led to Yang spending most of her nights sleeping in Blake's room on the floor until Blake caught her, and invited her into her bed. After that is was only a matter of time before their forced closeness turned into a relationship.

Yang shivered, pulling the blankets up closer to her. Winter was here and she was especially happy to have someone to share a bed with. She started to drift off again to memories of Blake keeping her warm when a knock at the door startled her awake.

"Uh oh." She said, remembering that Ruby said she was going to come by. Yang hadn't told Ruby (or anyone really) about her and Blake. She glanced over at the bed that was supposed to be hers in the corner, it was still freshly made from a week ago when she had changed the sheets to keep up appearances. Yang flew out of bed and to the dresser, hastily getting clothes on. She ran over to her bed and frantically messed it up before sprinting to the door. Yang stopped with her hand on the doorknob and collected herself. After a moment she swung the door open.<|endoftext|>Ethan Dean is on the verge of financial ruin because the city of Winona won't let him do what countless property owners have done for centuries: rent out his home.

Dean serves as an advisor in Iraq and Afghanistan and needs rental income to stay afloat while he's away. He never expected that answering the call to defend liberty abroad would lead to potential disaster at home because Winona does not respect traditional American property rights.

In America, renting one's property has always been considered a legitimate right of ownership. Winona, however, would rather see homeowners go broke than allow them to exercise it. The City Council has arbitrarily mandated that only 30 percent of the houses on any city block can be rented out. If 10 people live on your block, only three of your neighbors can obtain rental licenses. You and six other homeowners are forbidden from renting out your homes.

Dean lives on a block on which 30 percent of the houses have rental licenses; the ban prevents him from offsetting the cost of making mortgage payments on a house he does not live in.

Dean is not the only one affected by the city's ban.

Holly Richard and Ted and Lauren Dzierzbicki are three other homeowners suffering under this law. Richard attended school at St. Mary's University but has left to pursue a doctoral degree in South Dakota. The Dzierzbickis' daughter attended school at Winona State but has graduated, and the family wants to move on.

Dean, Richard and the Dzierzbickis each have decided to leave to pursue life, liberty and happiness. They put their Winona homes on the market, hoping to sell. Unfortunately, the down economy made that difficult to do at reasonable prices. Unwilling to sell at a serious loss, their next move was to rent out the homes and at least make their monthly mortgage payments. Renting is straight out of Financial Health 101 and is what all the popular financial pundits advise struggling homeowners to do.

But that's when these property owners ran up against the city's rental ban. The homes cannot be sold and they cannot be lived in. Dean and Richard are temporarily saved by a short-term exception, allowing them to rent their homes for now. But that exception expires in April 2012. The Dzierzbickis' house has stood empty for a year and a half -- all because the government stops them from doing what should be their right as property owners.

To make matters even more outrageous, the government doesn't even require that those holding one of the rental-home permits actually rent the house out.

This ban hurts not only homeowners but renters, too, because with fewer places available -- because of government-created scarcity in the market -- those fortunate few who have the government's blessing can drive up rents.

All this, however, is about to change if Dean, Richard and the Dzierbickis are successful. They have teamed up with the Institute for Justice, a public-interest law firm that protects property rights and economic liberty, to strike down this violation of their rights. Together they are filing a lawsuit challenging the law under the Minnesota Constitution.

The lawsuit has implications across the state and around the country. In Minnesota, Mankato, Northfield and West St. Paul have passed similar rental bans, while cities elsewhere, such as East Lansing, Mich., are imposing their own rental bans. Other cities are undoubtedly considering equally bad restrictions.

Property rights are about more than ownership. They are about being able to use your property in a way that makes the most sense for you and your family when life takes unexpected turns. Minnesotans had the foresight to enact a state Constitution that protects property rights, and it is time for Winona to learn that the purpose of government is to protect those rights, not to drive citizens to financial ruin.

Katelynn McBride is an attorney with the Institute for Justice Minnesota Chapter.<|endoftext|>Kanger SUBVOD Full Sub Ohm Kit

Made with, the Kanger SUBVOD is. Not only is it, the Kanger SUBVOD kitneeded for- not to mention it's a really good looking ecig,• 1300mAh integrated battery cell• 18mm diameter• Can fire 0.4Ω coils• Single button operation system• Micro USB charging port (Charges at 500mAh)• Passthrough technology (Can vape while the battery is being charged)• Numerous safety and protection systems• Threaded top filling system (Sealed with double o-rings)• Adjustable air flow to fit your preferred inhalation strength• 18mm diameter | 3.2ml Capacity | Pyrex glass tank• Configured for SSOCC and Clapton coil heads (Crisp and clear flavor delivery)• Gold plated 510 connection• Changeable 510 drip trip (Can use any drip tip you want)With the SUBVOD, Kanger brought together elements from several of their most successful ecigarettes. In addition to the, the SUBVOD kit features theand, the Toptank Nano is also equipped with anthat lets you choose how much air flows through the system, thereby allowing you to. Compact, lightweight and very attractive,<|endoftext|>Hundreds of supporters of deposed Egyptian President Mohamed Morsi remain trapped in a mosque near Cairo's Ramses Square after security forces escorted several of them out.

The standoff at the Egyptian capital's Fateh Mosque, which began late on Friday night, stretched into Saturday morning amid reports of offers of safe passage.

Speaking to Al Jazeera by phone from inside the mosque, Omaima Halawa said there were about 700 people, including women and children, inside and that they feared leaving the mosque because "there were thugs outside with the security forces, and that ... the security forces were working with the thugs".

She said she feared about what might happen to her or where she would be taken if she left the mosque.

Egypt's Nile News reported that about 10 people, mostly women, left the mosque accompanying the body of a woman who had died on Friday.

Al Jazeera Mubasher Misr said the protesters had been offered safe passage out of the mosque on the condition they would be subjected to investigation in an army camp.

The protesters rejected the conditions and insisted that any investigation be conducted inside the mosque, the TV station said.

For his part, the Egyptian army's spokesman said via Facebook that armed men had been shooting from the mosque at nearby buildings.

Violence erupted across Egypt again on Friday after the Muslim Brotherhood and other groups, under the banner of the Anti-Coup Alliance, called for protests in defiance of a security crackdown on pro-Morsi sit-ins that had left more than 600 dead since Wednesday.

At least 173 people were killed and 1,330 others were injured nationwide as protesters tried to stage Day of Rage marches against the military-led government, according to a government spokesman.

Police also arrested more than 1,000 suspected Muslim Brotherhood supporters, including 558 in Cairo alone, on Friday, the Interior Ministry said in a statement.

The Fateh Mosque had been turned into both a morgue and a field hospital by Morsi supporters until the standoff with security forces began.

In another development, Morsi'sFreedom and Justice Party (FJP) confirmed on Saturday that Ammar al-Badie, son of the Muslim Brotherhood's supreme guide, Mohamed al-Badie, was among those killed in Ramses Square in Friday's violence.

He was shot twice in the head and eyes, FJP said.

Badie is the third child of a high-ranking Brotherhood member to be killed this week. The other two died on Wednesday: Hafsa al-Shater, the daughter of the group's top strategist, Khairat al-Shater; and Asmaa el-Beltagy, the daughter of Mohamed el-Beltagy, the FJP head.

Live fire use authorised

The Muslim Brotherhood has called for a week of fresh demonstrations across Egypt following the latest deaths.

An interim cabinet, installed by the army after it removed Morsi during rallies against his rule, has refused to back down in the face of the ongoing protests.

It has authorised police to use live ammunition to defend themselves and state installations.

Bader Abdel Atty, a spokesman for the Egyptian Foreign Ministry, defended the actions of the security forces in an interview with Al Jazeera on Friday, saying that protesters were armed with machine guns.

"They are raising al-Qaeda flags in the heart of Cairo," he said.

"They are using machine guns against civilians. And this cannot be described as far as I know as a peaceful demonstration," he said.

Abdel Atty dismissed international condemnation of the violence and said Egypt would accept no external interference.

Many Western allies have condemned the killings, including the US, but Saudi Arabia threw its weight behind the Egyptian government on Friday, accusing the Muslim Brotherhood of trying to destabilise country.

However, with no compromise in sight, the most populous Arab nation - which is often seen as leading events in the entire region - looks increasingly polarised and angry.

A number of tour operators have suspended all holidays to Egypt until at least next month and the US has urged its citizens to leave the country.

The EU has asked its states to consider "appropriate measures" to take in reaction to the violence, while Germany said it was reconsidering its ties.<|endoftext|>– When Cara Jones let her black-and-white pit bull, “Creature,” out Thursday night to relieve herself at the end of William Street, she heard something unusual coming from the woods. “I kept hearing sticks breaking,” said Jones, 23, a hairstylist who works in Bridgewater. “I thought it was a fox.” Suddenly, her 2-year-old dog began barking wildly. “She was standing there, barking and looking back at me,” Jones said. Jones ran back into her home, grabbed a flashlight and cellphone and ran into the woods. There, in the brush she spotted 89-year-old Carmen Mitchell, lying on the ground. “She was covered in mud and she was barefoot,” Jones said.

The elderly woman had wandered from her family’s home in Piscataway earlier in the day, police said. The 5-foot, 7-inch woman was last seen hours earlier carrying her slippers near the Piscataway-Middlesex border, according to a report. Police searched for hours in patrol cars, on foot and with the help of a low-flying state police helicopter. But it was “Creature” who found the woman. “We had people all over the area,” said Capt. Kenneth Blair of the Piscataway Police Department. “We had fire units from every district; 10 police officers, state police helicopter. “But it was her dog who found her. The dog actually led (Jones) to the spot,” he said. Emergency rescue workers arrived on the scene almost immediately after Jones called. “I’ve never seen police respond so fast,” she said.

CONNECT WITH US

• Follow us on Twitter

• Like us on Facebook

• NJ.com/middlesex

Ambulance workers placed Mitchell on a stretcher and carried her out of the woods. She was taken to a local hospital for observation. The woman suffered slightly from hypothermia, said Blair, but other than that appeared to be OK. Jones said Friday morning she wanted the story about her dog to get out because it seemed the media had paid too much attention to the police helicopter that was used. “Pit bulls never get any recognition for doing something good,” she said. “Here’s a chance to say something good about a pit bull.”<|endoftext|>The first time I heard of this still-without-US-distribution German sci-fi indie was casually browsing through Facebook and the image of a scantily clad teenage girl protectively clutching to a little extraterrestrial in neon-fluorescent blue and red colors was destined to get my attention. After finally tracking down and viewing the Euro-music video infused “student film” however, I’m quickly realizing it’s better to keep scrolling past numbers like these on social media no matter how inviting the posters and premise appear to be.

Before unveiling the opening title cards, a series of inter-titles warn the viewer of stroboscopic effects potentially causing seizures in some viewers appears before then suggesting that the film be played loud. Starting out by ripping off of Gaspar Noe’swith its epileptic foreshadowing before closing on copying Abel Ferrara’s, the cinematic debut of visual artist AKIZ (Achim Bornhak) and reportedly the first entry in a plannedis among the most shamelessknockoffs to surface since. It is also arguably the most aggravating look-at-me demonstration of “experimental visual technique” since Jonas Ackerlund’s misbegotten

While many are quick to read Der Nachtmahr (The Nightmare) as a teenage coming of age science fiction horror psychodrama about a young girl who after a hallucinogen filled rave party starts to see a Gollum like creature monkeying about her home, I saw a wannabe eager to display the influences hanging far from it’s sleeves. When it isn’t blatantly stealing high watermarks from Spielberg’s 1982 classic, right down to the girl and creature feeling each other’s feelings with the Gollum being poked and prodded in plastic wrapped hospital rooms, it gives neither the poor girl nor her special friend much of anything to do but muck about in back alleyways or raiding the fridge leaving a mess echoing (again Spielberg’s) Close Encounters of the Third Kind .

received special attention for being made in Germany without the support of public broadcasting or film financing groups as well as going the Dogme 95 route of using natural lighting and set pieces without much additional dressing. Fine and dandy, but the results here when they don’t look like a YouTube video with many blocky looking nighttime scenes like Ackerlund’s aforementioned misfire don’t have anything substantive to say about the experimental techniques being used other than ‘aren’t they cool?’.also arrives on the heels of the equally troubling yet comparatively infinitely better coming of age teen shock-horror fest, a film I don’t necessarily recommend either but will take over the prospect ofa second time.

Every now and again a unique cinematic diamond-in-the-rough appears out of nowhere that can’t help but draw out my insect-like cinephile antennae searching for something truly unique. From the outset, Der Nachtmahr appeared to be this experimental and surrealist cinema junkie’s cup of tea. Upon actually sitting through it, I watched so much promise for offbeat provocation, technical innovation and sensory assault get all but completely squandered. It isn’t so much that it nakedly rips off all of the right people, which it does seemingly with relish in scene after scene. My problem with Der Nachtmahr is that once all the pieces taken from various sources are all in place, the film ultimately does absolutely nothing new with them.<|endoftext|>newsheadlines ROYAL BLOOD SPORTS

A few months before the Queen's German husband, Prince Philip, launched the World Wildlife Fund in 1961, he went on a tiger hunt with the Queen. A tiger was lured into range by tethered goats and shot dead by Philip who calls himself a conservationist and environmentalist.

On the same tour, this time in Kathmandu, Philip was in a shooting party with Alec Douglas Hume (Lord Home), the Conservative Prime Minister, Bilderberg Group chairman and bloodline of the Scottish Brotherhood families.

The World Wide Fund For Nature (WWF) was not created to save endangered species. Ian MacPhail, the WWF’s first international appeals director, told a British television crew how a mother elephant and her calf came into range. Philip shot the mother while her calf ran off in terror. MacPhail said he helped to cover up the incident because the WWF was about to be launched and he believed the Fund would benefit wildlife conservation.

It has always mystified the public to see the contradiction between Philip, the founder and driving force behind the WWF, and Philip the killer of animals and birds for the sheer enjoyment of it.

The Queen's husband, Prince Philip, and former Nazi SS officer Prince Bernhard of the Netherlands co‑founded the World Wildlife Fund (WWF) in 1961. The WWF has raised huge sums of money from multinational corporations like BP‚ Shell‚ Rio Tinto‚ and Unilever to lock up great tracts of land, first in Africa, and then all over the world‚ so that the land could not be used by locals for economic development of their nations to raise their standard of living.

GREEN GENOCIDE Philip's "mean green" movement is selling "bad science" on climate-change which is designed to create food shortages, farming shut-downs and population dependency. It is part of the New World Order one-world-government agenda strategy. The WWF is a vehicle for controlling wildlife parks in Africa and elsewhere in which terrorist groups and mercenaries can gather, train, and cross borders to bring genocide to places like Rwanda and Burundi.

The WWF coordinates and funds the systematic slaughter of people and animals and has made a fortune from the illegal trade in ivory it was supposed to be trying to stop. To impose global ‘solutions’, the elite need global ‘problems’... and the environment is perfect for that. It allows them to pass international laws and create centralized, global organizations to police them. It allows them to move native peoples from their ancient lands to create wildlife parks and ‘conservation’ areas all over the world, particularly in Africa and the Americas, which then come under the centralized control of the elites. It gives them footholds in strategic areas where they can launch ‘freedom fighters’ to start civil wars.

FOOD AS A WEAPON Humanity’s future is at stake. Green policies have been used by the British oligarchy to de-industrialise nations and to smash their agricultural production. This will result in a world food shortage and starvation. The WWF is one of the largest contributors to depopulation efforts worldwide. Ten to twelve companies run the world’s food supply which are grouped around Britain’s Royal House of Windsor. Led by the six leading grain companies—Cargill, Continental, Louis Dreyfus, Bunge and Born, André, and Archer Daniels Midland/Töpfer—the Windsor-led food and raw materials cartel has complete domination over world cereals and grains supplies, from wheat to corn and oats, from barley to sorghum and rye. But it also controls meat, dairy, edible oils and fats, fruits and vegetables, sugar, and all forms of spices. HOARDING FOOD The elites view themselves as inheritors of the Earth. Eugenics is their answer to freeing-up Planet Earth's dwindling resources that are being consumed by "racially inferior stocks" within the human gene pool. The oligarchy is "hoarding" its food and raw materials holdings...and they are prepared to shut down food production and export supplies, not only to poor nations, but to advanced sector nations as well. Today, food warfare is firmly under the control of London, with the help of subordinate partners. The Windsor-led oligarchy has built up a single, integrated raw materials cartel, with three divisions—energy, raw materials and minerals, and increasingly scarce food supplies.

The queens German husband, Prince Philip Mountbatten, alias Philip Battenberg has been quoted as saying: "In the event that I am reincarnated, I would like to return as a deadly virus to solve the overpopulation problem". The late Jacques Cousteau, a famed advocate for the oceans and the environment, was quoted in a November 1991 UNESCO Courier saying, "In order to stabilize world populations, we must eliminate 350,000 people per day. It is a horrible thing to say, but it's just as bad not to say it."

THE NAZI CONNECTION World Wildlife Fund co-founders, Philip and Bernhard, are from the same bloodline. Philip and his royal family are steeped in Nazi connections. Bernhard was a member of Himmler’s murderous SS. He was born a German in 1912, the cousin-in-law of Princess Victoria of Hohenzollern, the sister of Kaiser Wilhelm. He was recruited into Nazi Intelligence at the University of Berlin in 1934 and worked for the SS operation within I. G. Farben, the chemical giant which had such close connections with the Rockefeller/ Farish Standard Oil and British companies like ICI.

Bernhard’s background caused a scandal in the Netherlands when he married Queen Juliana of the infamous House of Orange, to become the Netherlands equivalent of his bosom buddy, Prince Philip. Bernhard helped to found the Bilderberg Group which officially met for the first time in 1954, and in 1961 he co-founded, with Philip, the World Wildlife Fund (now the World Wide Fund For Nature), funded in part by the Mellons.
"""

with open("./output/bpe_owt.pkl", "rb") as f:
    owt_bpe = pickle.load(f)

owt_special_tokens = tinystories_special_tokens.copy()

owt_tokenizer = Tokenizer(owt_bpe["vocab"], owt_bpe["merges"], owt_special_tokens)
owt_encoding = owt_tokenizer.encode(owt_text)

print("OWT Size")
encoding_size = len(owt_encoding)*16
print(f"Encoding size {encoding_size}")
text_size = len(owt_text.encode())
print(f"Original size {text_size*8}")

# %%
print("Throughput estimation")
with open("../data/owt_valid.txt", "r") as f:
    owt_read = f.read()[0:1000000]

start = time.time()
tokens = owt_tokenizer.encode(owt_read)
end = time.time()

elapsed = end - start
num_chars = len(owt_read)
num_tokens = len(tokens)

print(f"Chars encoded: {num_chars}")
print(f"Tokens produced: {num_tokens}")
print(f"Elapsed time: {elapsed:.6f}s")
print(f"Throughput (chars/sec): {num_chars / elapsed:,.2f}")
print(f"Throughput (tokens/sec): {num_tokens / elapsed:,.2f}")

# %%
