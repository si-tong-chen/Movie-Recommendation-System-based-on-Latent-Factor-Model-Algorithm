import numpy as np
import tensorflow as tf
import random
import io
import sys

iter = []
train_loss,train_mae=[],[]
valid_loss, valid_mae=[],[]

train_iter,validation_iter = [],[]

train_loss_batch_record, train_mae_batch_record=[],[]
valid_loss_batch_record, valid_mae_batch_record=[],[]



class SVDModel(tf.keras.Model):
    '''
    define LFM model
    '''

    def __init__(self, user_num, item_num, dim,reg_strength):
        super(SVDModel, self).__init__()
        self.reg_strength = reg_strength  
        initializer = tf.keras.initializers.TruncatedNormal(stddev=0.02)
        self.user_emb = tf.keras.layers.Embedding(user_num, dim, embeddings_initializer=initializer,trainable=True)
        self.item_emb = tf.keras.layers.Embedding(item_num, dim, embeddings_initializer=initializer,trainable=True)
                                
        self.global_bias = tf.Variable(initial_value=0.0, dtype=tf.float32, name="global_bias")
        self.bias_user = tf.keras.layers.Embedding(user_num, 1, embeddings_initializer='zeros', name="bias_user")
        self.bias_item = tf.keras.layers.Embedding(item_num, 1, embeddings_initializer='zeros', name="bias_item")
    
    def call(self, inputs):
        user, item = inputs
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        
        dot = tf.reduce_sum(tf.multiply(user_emb, item_emb), axis=1)
        
        global_bias = self.global_bias
        bias_user = self.bias_user(user)
        bias_item = self.bias_item(item)
        
        output = dot + global_bias + tf.transpose(bias_user)+tf.transpose(bias_item)
      
        

        output_star = tf.clip_by_value(output, 1.0, 5.0)
        
        reg_loss = self.reg_strength * (tf.reduce_sum(tf.square(user_emb)) + tf.reduce_sum(tf.square(item_emb))) 
        self.add_loss(reg_loss)

        return output_star
        



class SearchPara:
    '''
    the aim of this part to search parameters.the workfolw is almost same as the LFM(in the script)
    so we don't explain the code in here. we have expained in the LFM.
    the difference is  we calculate each epoch validation loss(the mean of all the batch) as the 
    judgment criteria to find the best parameters.
    '''
    def __init__(self,num_epochs,num_searches,dim_values,
                 reg_strength_values,learning_rate_values,
                 clipvalue_values,item_num,user_num,train_data,validation_data,batch_size):
        
        self.num_epochs=num_epochs
        self.num_searches=num_searches
        self.dim_values=dim_values
        self.reg_strength_values=reg_strength_values
        self.learning_rate_values=learning_rate_values
        self.clipvalue_values=clipvalue_values
        self.user_num =user_num
        self.item_num =item_num
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size=batch_size
        self.best_hyperparameters = None 
        self.best_val_loss = float('inf') 
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError()


    @tf.function
    def train_step(self,users, items, rates):
        with tf.device('/GPU:0'):  # 选择第一个GPU
            with tf.GradientTape() as tape:
                users= tf.convert_to_tensor(users)
                items = tf.convert_to_tensor(items)
                rates =tf.convert_to_tensor(rates)
                inputs = (users, items)
                output_star = self.model(inputs, training=True)
                loss = self.loss_object(rates, output_star)
                total_loss = loss + tf.reduce_sum(self.model.losses)       

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            self.mae_metric.update_state(rates, output_star)
            mae = self.mae_metric.result()

        return total_loss, mae

    @tf.function 
    def validation_step(self,users, items, rates):
        users= tf.convert_to_tensor(users)
        items = tf.convert_to_tensor(items)
        rates =tf.convert_to_tensor(rates)

        inputs = (users, items)
        self.model.trainable = False  # 关闭模型不让更新
        output_star = self.model(inputs, training=False)
        val_loss = self.loss_object(rates, output_star)

        self.mae_metric.update_state(rates, output_star)
        val_mae = self.mae_metric.result()

        return val_loss, val_mae

    def search(self):
        output_buffer = io.StringIO()
        sys.stdout = output_buffer
        
        for num in range(self.num_searches):
            dim = random.choice(self.dim_values)
            reg_strength = random.choice(self.reg_strength_values)
            learning_rate = random.choice(self.learning_rate_values)
            clipvalue= random.choice(self.clipvalue_values)
            self.model = SVDModel(self.user_num, self.item_num, dim,reg_strength)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate, clipvalue= clipvalue)


            for epoch in range(self.num_epochs):

                self.mae_metric.reset_states()  
                train_loss_batch, train_mae_batch= [], [] 

                for i, (users, items, rates) in enumerate(self.train_data):
                    loss, mae = self.train_step(users, items, rates)
                    train_loss_batch.append(loss.numpy())
                    train_mae_batch.append(mae.numpy())                
                    train_loss_batch_record.append(loss.numpy())
                    train_mae_batch_record.append(mae.numpy())
                    train_iter.append(i+1)
                    if i+1 >= self.batch_size:
                        break

                train_loss.append(np.mean(train_loss_batch))
                train_mae.append(np.mean(train_mae_batch))

                valid_loss_batch, valid_mae_batch = [],[]

                for j, (users, items, rates) in enumerate(self.validation_data):
                    val_loss, val_mae = self.validation_step(users, items, rates)
                    valid_loss_batch.append(val_loss.numpy())
                    valid_mae_batch.append(val_mae.numpy())                   
                    valid_loss_batch_record.append(val_loss.numpy())
                    valid_mae_batch_record.append(val_mae.numpy())            
                    validation_iter.append(j+1)

                    if j+1 >= self.batch_size:
                        break
                


                valid_loss.append(np.mean(valid_loss_batch))
                valid_mae.append(np.mean(valid_mae_batch))

                iter.append(epoch+1)
                average_val_loss = np.mean(valid_loss)

            print(f"num_searches {num+1}: average_val_loss = {average_val_loss}")

            if average_val_loss < self.best_val_loss:
                    self.best_val_loss = average_val_loss
                    self.best_hyperparameters = {
                            'dim': dim,
                            'reg_strength': reg_strength,
                            'learning_rate': learning_rate
                        }


        
        print("best_hyperparameters：", self.best_hyperparameters)
        print("best_val_loss：",self.best_val_loss)


        sys.stdout = sys.__stdout__


        all_print_output = output_buffer.getvalue()
        
        output_buffer.close()

     
        return all_print_output

                
