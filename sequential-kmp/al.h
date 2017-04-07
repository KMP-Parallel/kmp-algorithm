#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
//#define ARRAYLIST_INITIAL_CAPACITY 4

typedef struct ArrayList ArrayList;
struct ArrayList {
    int size; // the count of elements in arralylist
    int* body;
    int start;
};


int get(ArrayList list, int idx);
ArrayList add(ArrayList list, int value);
ArrayList set(ArrayList list, int index, int value);
ArrayList initial_arr(int size);
void printTable(ArrayList list);

ArrayList initial_arr(int size){
    ArrayList table;
    table.size = size;
    table.start = 0;
    table.body = (int *)malloc(size * sizeof(int));
    return table;
}

int get(ArrayList list, int idx){
    assert(list.size > idx);   
    return list.body[idx];
}
ArrayList add(ArrayList list,int value){
    list.body[list.start++] = value;
    return list;
}

ArrayList set(ArrayList list, int index, int value){
    list.body[index] = value;
    return list;
}

void printTable(ArrayList list){
    for(int i=0;i<list.size;i++)
        printf("%d ", list.body[i]);
    printf("\n");
}